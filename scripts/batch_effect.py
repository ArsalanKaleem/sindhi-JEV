"""How much do reported AUROCs depend on batch padding? Synthetic test sets only (CPU).

    python scripts/batch_effect.py

Batched = PyTorch, batch 64 (the setting of every reported number; cached in runs/main_v2/preds_<set>.json).
Single = the ONNX tool, one crop per forward pass (no padding). Finding of 2026-11-26: BatchNorm makes padded
columns non-zero and convolutions carry them into the last real tokens, so a score depends on its batch.
Writes results/batch_effect.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from sindhijev.metrics import auroc  # noqa: E402
from sindhijev.tool import Checker  # noqa: E402
from ablation_table import scores  # noqa: E402

SETS = ["test_iid", "test_font", "test_hard", "test_font_hard"]


def main():
    ck = Checker(ROOT / "dist" / "sindhijev_v2.onnx")
    res = {}
    for s in SETS:
        ref = scores("main_v2", s, torch.device("cpu"))
        rows = [json.loads(x) for x in open(ROOT / "data" / "synth" / s / "manifest.jsonl", encoding="utf-8")]
        y, zb, zs = [], [], []
        for r in rows:
            p = ck.check(ROOT / "data" / "synth" / s / "scan" / f"{r['id']}.png", r["cand"])["p_wrong"]
            pb = min(max(ref[r["id"]][1], 1e-7), 1 - 1e-7)
            y.append(ref[r["id"]][0])
            zs.append(np.log(p / (1 - p)) * ck.temperature)
            zb.append(np.log(pb / (1 - pb)))
        d = np.abs(np.array(zs) - np.array(zb))
        res[s] = {"n": len(rows), "auroc_batched": auroc(y, zb), "auroc_single": auroc(y, zs),
                  "logit_diff_median": float(np.median(d)), "logit_diff_p95": float(np.percentile(d, 95)),
                  "logit_diff_max": float(d.max())}
        print(s, {k: round(v, 4) for k, v in res[s].items()}, flush=True)
    (ROOT / "results" / "batch_effect.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
