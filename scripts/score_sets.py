"""Quick AUROC + char metrics of one checkpoint on chosen synthetic sets (CPU-friendly; no bootstrap).

    python scripts/score_sets.py --ckpt runs/main/last.pt --sets val_font val_font_hard --out results/v1_on_val_font.json

Used for the v1-vs-v2 rule (2026-11-22): v2 replaces v1 only if its val_font_hard AUROC is higher.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from sindhijev.data import Manifest, collate  # noqa: E402
from sindhijev.metrics import auroc, char_metrics  # noqa: E402
from evaluate import load_model  # noqa: E402
from train import predict  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--char-thr", type=float, default=None, help="default: the value stored in results/synth_main.json")
    a = ap.parse_args()
    dev = torch.device("cpu")
    model, st = load_model(a.ckpt, dev)
    thr = a.char_thr
    if thr is None:
        thr = json.loads((ROOT / "results" / "synth_main.json").read_text(encoding="utf-8"))["char_threshold_from_val"]
    res = {"ckpt": a.ckpt, "step": st["step"], "char_thr": thr}
    for s in a.sets:
        ds = Manifest(ROOT / "data" / "synth" / s, model.cfg)
        recs = predict(model, DataLoader(ds, batch_size=64, collate_fn=collate, num_workers=0), dev, False)
        res[s] = {"n": len(recs), "auroc": auroc([r["seq_y"] for r in recs], [r["seq_score"] for r in recs]),
                  **char_metrics(recs, thr)}
        print(s, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[s].items()}, flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
