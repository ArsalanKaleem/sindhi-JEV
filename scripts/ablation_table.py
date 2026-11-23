"""Ablation table: each run vs abl_ref (same 16k-step budget), paired clustered bootstrap on the same examples (CPU).

    python scripts/ablation_table.py --boot 1000

Needs runs/abl_*/last.pt and results/synth_abl_*.json (from scripts/evaluate.py, for char F1 / top-1).
Sequence scores per run and set are cached in runs/<run>/preds_<set>.json (not tracked).
Writes results/ablations.json and results/ablations.md. Runs are compared, never selected for use.
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
from sindhijev.metrics import auroc, paired_bootstrap_delta  # noqa: E402
from evaluate import load_model  # noqa: E402
from train import predict  # noqa: E402

RUNS = [("abl_ref", "nothing (reference)"), ("abl_noscan", "scan image"), ("abl_norender", "re-rendered candidate"),
        ("abl_noalign", "char-aligned render pooling"), ("abl_nocharhead", "per-character head and loss"),
        ("abl_v1mix", "real-word swaps in training (v1 error mix)")]
SETS = ["val_hard", "val_font_hard", "test_font", "test_hard", "test_font_hard"]


def scores(run, s, dev):
    cache = ROOT / "runs" / run / f"preds_{s}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    model, _ = load_model(ROOT / "runs" / run / "last.pt", dev)
    ds = Manifest(ROOT / "data" / "synth" / s, model.cfg)
    recs = predict(model, DataLoader(ds, batch_size=64, collate_fn=collate, num_workers=0), dev, False)
    out = {r["id"]: [r["seq_y"], r["seq_score"]] for r in recs}
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    dev = torch.device("cpu")
    res = {"sets": SETS, "boot": a.boot, "runs": {}}
    for run, removed in RUNS:
        synth = json.loads((ROOT / "results" / f"synth_{run}.json").read_text(encoding="utf-8"))
        res["runs"][run] = {"removed": removed, "step": synth["step"]}
        for s in SETS:
            ref, cur = scores("abl_ref", s, dev), scores(run, s, dev)
            assert ref.keys() == cur.keys()
            recs = [{"seq_y": cur[k][0], "a": cur[k][1], "b": ref[k][1], "cluster": k} for k in cur]
            o = {"auroc": auroc([r["seq_y"] for r in recs], [r["a"] for r in recs]),
                 "char_f1": synth[s].get("char_f1"), "top1_loc": synth[s].get("top1_loc")}
            if run != "abl_ref":
                o["vs_ref"] = paired_bootstrap_delta(recs, "a", "b", n=a.boot)
            res["runs"][run][s] = o
            print(run, s, round(o["auroc"], 4), o.get("vs_ref", {}).get("ci"), flush=True)
    (ROOT / "results" / "ablations.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    hdr = " | ".join(SETS)
    lines = ["# Ablations (synthetic, 16k steps each, last.pt)", "",
             "AUROC, and in brackets the change vs `abl_ref` with its paired clustered-bootstrap 95% CI "
             f"({a.boot} resamples). Letter F1 on test_font_hard. `abl_nocharhead` has no per-letter output.", "",
             f"| run | removed | {hdr} | letter F1 |", "|---|---|" + "---|" * len(SETS) + "---|"]
    for run, removed in RUNS:
        r = res["runs"][run]
        cells = []
        for s in SETS:
            v = r[s].get("vs_ref")
            cells.append(f"{r[s]['auroc']:.4f}" + (f" ({v['delta']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}])"
                                                    if v else ""))
        f1 = "n/a" if run == "abl_nocharhead" else f"{r['test_font_hard']['char_f1']:.3f}"
        lines.append(f"| `{run}` | {removed} | " + " | ".join(cells) + f" | {f1} |")
    (ROOT / "results" / "ablations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
