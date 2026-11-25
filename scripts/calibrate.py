"""Temperature scaling for the tool's p_wrong: fitted on SYNTHETIC val + val_hard only, checked on synthetic test sets.

    python scripts/calibrate.py --run main_v2

T minimises the negative log-likelihood of sigmoid(logit / T) on val + val_hard (grid 0.25..4). AUROC is unchanged
by construction (monotone). Writes results/calibration_<run>.json. The real set is never used (issue #7: real ECE is
reported by eval_real.py, not tuned on).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from sindhijev.metrics import ece  # noqa: E402
from ablation_table import scores  # noqa: E402

FIT = ["val", "val_hard"]
CHECK = ["test_iid", "test_font", "test_hard", "test_font_hard"]


def logits(run, s):
    d = scores(run, s, torch.device("cpu"))
    y = np.array([v[0] for v in d.values()], float)
    p = np.clip(np.array([v[1] for v in d.values()], float), 1e-7, 1 - 1e-7)
    return y, np.log(p / (1 - p))


def nll(y, z, t):
    p = np.clip(1 / (1 + np.exp(-z / t)), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main_v2")
    a = ap.parse_args()
    fit = [logits(a.run, s) for s in FIT]
    y, z = np.concatenate([f[0] for f in fit]), np.concatenate([f[1] for f in fit])
    grid = np.exp(np.linspace(math.log(0.25), math.log(4), 400))
    t = float(grid[int(np.argmin([nll(y, z, g) for g in grid]))])
    res = {"run": a.run, "fit_sets": FIT, "temperature": t, "fit_nll": {"T=1": nll(y, z, 1.0), "T": nll(y, z, t)},
           "ece": {}}
    for s in FIT + CHECK:
        ys, zs = logits(a.run, s)
        res["ece"][s] = {"T=1": ece(ys.tolist(), (1 / (1 + np.exp(-zs))).tolist()),
                         "T": ece(ys.tolist(), (1 / (1 + np.exp(-zs / t))).tolist())}
        print(s, {k: round(v, 4) for k, v in res["ece"][s].items()}, flush=True)
    print("temperature", round(t, 3))
    (ROOT / "results" / f"calibration_{a.run}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
