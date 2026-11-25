"""Export a checkpoint to ONNX for the CPU tool and check it against PyTorch on synthetic examples.

    python scripts/export_onnx.py --ckpt runs/main_v2/last.pt --out dist/sindhijev_v2.onnx

Inputs (batch B, any widths/lengths): scan [B,1,48,Ws] f32, scan_w [B] i64, cand [B,1,48,Wc] f32, cand_w [B] i64,
ids [B,L] i64, char_x [B,L,2] f32. Outputs: seq_logit [B], char_logit [B,L].
Parity check: max |logit difference| on 16 val_hard examples (batched, padded) must be < 1e-3.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from sindhijev.data import Manifest, collate  # noqa: E402
from evaluate import load_model  # noqa: E402

NAMES = ["scan", "scan_w", "cand", "cand_w", "ids", "char_x"]


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, scan, scan_w, cand, cand_w, ids, char_x):
        o = self.m({"scan": scan, "scan_w": scan_w, "cand": cand, "cand_w": cand_w, "ids": ids, "char_x": char_x})
        return o["seq_logit"], o["char_logit"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/main_v2/last.pt")
    ap.add_argument("--out", default="dist/sindhijev_v2.onnx")
    ap.add_argument("--name", default="main_v2", help="results/synth_NAME.json, calibration_NAME.json")
    a = ap.parse_args()
    model, st = load_model(ROOT / a.ckpt, torch.device("cpu"))
    w = Wrap(model).eval()
    ds = Manifest(ROOT / "data" / "synth" / "val_hard", model.cfg, limit=16)
    b = collate([ds[i] for i in range(len(ds))])
    args = tuple(b[k] for k in NAMES)
    B, Ws, Wc, L = "B", "Ws", "Wc", "L"
    dyn = {"scan": {0: B, 3: Ws}, "scan_w": {0: B}, "cand": {0: B, 3: Wc}, "cand_w": {0: B}, "ids": {0: B, 1: L},
           "char_x": {0: B, 1: L}}
    out = ROOT / a.out
    out.parent.mkdir(exist_ok=True)
    torch.backends.mha.set_fastpath_enabled(False)  # fused MHA kernel has no ONNX symbolic; same math
    with torch.no_grad():
        torch.onnx.export(w, args, str(out), input_names=NAMES, output_names=["seq_logit", "char_logit"],
                          dynamic_axes=dyn, opset_version=18, dynamo=True)  # legacy exporter fixed L (2026-11-26)
    import onnx  # noqa: PLC0415
    data = out.with_name(out.name + ".data")
    onnx.save(onnx.load(str(out)), str(out), save_as_external_data=False)  # one self-contained file (~29 MB)
    data.unlink(missing_ok=True)
    import onnxruntime as ort  # noqa: PLC0415
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    res = {"ckpt": a.ckpt, "step": st["step"]}
    for name, bb in [("batch16", b), ("single", collate([ds[3]]))]:
        feeds = {k: bb[k].numpy() for k in NAMES}
        got = sess.run(None, feeds)
        with torch.no_grad():
            r = w(*(bb[k] for k in NAMES))
        mask = bb["ids"].numpy() != 0
        res[name] = {"seq_max_abs": float(np.abs(got[0] - r[0].numpy()).max()),
                     "char_max_abs": float(np.abs(got[1] - r[1].numpy())[mask].max())}
    print(json.dumps(res, indent=1), f"\n{out}: {out.stat().st_size / 1e6:.1f} MB")
    assert all(v < 1e-3 for k in ("batch16", "single") for v in res[k].values()), "ONNX does not match PyTorch"
    # model card read by sindhijev.tool: temperature (scripts/calibrate.py) and char threshold, both from synthetic val
    cal = json.loads((ROOT / "results" / f"calibration_{a.name}.json").read_text(encoding="utf-8"))
    syn = json.loads((ROOT / "results" / f"synth_{a.name}.json").read_text(encoding="utf-8"))
    card = {**res, "model_cfg": st["model_cfg"], "temperature": cal["temperature"],
            "char_threshold": syn["char_threshold_from_val"], "source": "https://github.com/ArsalanKaleem/sindhi-JEV",
            "license": "Apache-2.0"}
    (out.with_suffix(".json")).write_text(json.dumps(card, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
