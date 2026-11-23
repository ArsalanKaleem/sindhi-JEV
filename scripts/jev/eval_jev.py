"""Score Jev zero-shot outputs on synthetic sets against labels, and vs the v2 model on the same examples (CPU).

    python scripts/jev/eval_jev.py --results runs/jev        # expects runs/jev/jev_<set>.jsonl

Jev returns P(candidate correct); the error score is 1 - prob_1, so AUROC is comparable with our seq_score.
Failed items (prob_1 null, 0.0 or -1.0 with an error) are counted and scored as 0.5 (no information), never dropped.
Writes results/synth_jev.json and results/synth_jev.md.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from sindhijev.metrics import auroc, paired_bootstrap_delta  # noqa: E402
from ablation_table import scores  # noqa: E402

SETS = ["test_hard", "test_font_hard"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="runs/jev")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    meta = json.loads((ROOT / a.results / "run_info.json").read_text(encoding="utf-8")) \
        if (ROOT / a.results / "run_info.json").exists() else {}
    res = {"run_info": meta}
    lines = ["# Jev zero-shot on synthetic sets", "",
             f"MiniCPM-OCR Jev as released, score-only path with our renders; run info: `{json.dumps(meta)}`", "",
             "| set | n | failed | Jev AUROC | v2 AUROC | v2 - Jev [95% CI] |", "|---|---|---|---|---|---|"]
    for s in SETS:
        p = ROOT / a.results / f"jev_{s}.jsonl"
        if not p.exists():
            continue
        jev = {}
        failed = 0
        for x in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(x)
            ok = r["prob_1"] is not None and not r.get("error") and 0.0 <= r["prob_1"] <= 1.0
            failed += not ok
            jev[r["id"]] = 1 - r["prob_1"] if ok else 0.5
        v2 = scores("main_v2", s, torch.device("cpu"))
        ids = [k for k in v2 if k in jev]
        recs = [{"seq_y": v2[k][0], "a": v2[k][1], "b": jev[k], "cluster": k} for k in ids]
        y = [r["seq_y"] for r in recs]
        o = {"n": len(recs), "n_set": len(v2), "failed": failed, "jev_auroc": auroc(y, [r["b"] for r in recs]),
             "v2_auroc": auroc(y, [r["a"] for r in recs]),
             "v2_vs_jev": paired_bootstrap_delta(recs, "a", "b", n=a.boot)}
        res[s] = o
        d = o["v2_vs_jev"]
        lines.append(f"| {s} | {o['n']}/{o['n_set']} | {failed} | {o['jev_auroc']:.4f} | {o['v2_auroc']:.4f} | "
                     f"{d['delta']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}] |")
        print(lines[-1], flush=True)
    (ROOT / "results" / "synth_jev.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (ROOT / "results" / "synth_jev.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
