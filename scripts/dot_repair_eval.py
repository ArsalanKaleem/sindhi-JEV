"""Dot repair on synthetic sets: choose the margin on val_hard, report it unchanged on the test sets (ONNX, CPU).

    python scripts/dot_repair_eval.py --weights dist/sindhijev_v2.onnx

Repair = Checker.repair (top-3 suspicious letters, reviewed dot partners, up to 2 swaps, keep a swap only if p_wrong
drops by >= margin). One greedy pass with margin 0 records each step's drop; any margin m keeps the steps up to the
first drop < m, so all margins come from one run. Reported per set: exact match before/after, errors fixed,
correct texts broken. Writes results/dot_repair.json and .md. The real set is not used.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sindhijev.realset import norm  # noqa: E402
from sindhijev.tool import Checker  # noqa: E402

MARGINS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def run_set(ck, name, limit=None):
    d = ROOT / "data" / "synth" / name
    rows = [json.loads(x) for x in open(d / "manifest.jsonl", encoding="utf-8")][:limit]
    out = []
    for i, r in enumerate(rows):
        _, hist = ck.repair(d / "scan" / f"{r['id']}.png", r["cand"], margin=1e-9)
        out.append({"id": r["id"], "true": norm(r["true"]), "cand": norm(r["cand"]), "mode": r["mode"],
                    "steps": [(h["to"], h["p_wrong"][0] - h["p_wrong"][1]) for h in hist]})
        if (i + 1) % 500 == 0:
            print(name, i + 1, flush=True)
    return out


def at_margin(recs, m):
    n = len(recs)
    before = after = fixed = broken = 0
    for r in recs:
        final = r["cand"]
        for to, drop in r["steps"]:
            if drop < m:
                break
            final = to
        ok0, ok1 = r["cand"] == r["true"], final == r["true"]
        before += ok0
        after += ok1
        fixed += (not ok0) and ok1
        broken += ok0 and not ok1
    return {"exact_before": before / n, "exact_after": after / n, "fixed": fixed, "broken": broken, "n": n,
            "n_wrong": sum(r["cand"] != r["true"] for r in recs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="dist/sindhijev_v2.onnx")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    ck = Checker(ROOT / a.weights)
    val = run_set(ck, "val_hard", a.limit)
    sweep = {m: at_margin(val, m) for m in MARGINS}
    best = max(MARGINS, key=lambda m: (sweep[m]["fixed"] - sweep[m]["broken"], m))
    res = {"weights": a.weights, "chosen_on": "val_hard", "margin": best, "val_hard_sweep": sweep, "test": {}}
    lines = ["# Dot repair (synthetic)", "",
             f"Margin {best} chosen on val_hard (max fixed - broken); test sets use it unchanged.", "",
             "| set | n | wrong before | exact before | exact after | fixed | broken |", "|---|---|---|---|---|---|---|"]
    for s in ["test_hard", "test_font_hard", "test_iid", "test_font"]:
        o = at_margin(run_set(ck, s, a.limit), best)
        res["test"][s] = o
        lines.append(f"| {s} | {o['n']} | {o['n_wrong']} | {o['exact_before']:.3f} | {o['exact_after']:.3f} | "
                     f"{o['fixed']} | {o['broken']} |")
        print(lines[-1], flush=True)
    lines += ["", "val_hard sweep: " + ", ".join(f"m={m}: +{v['fixed']}/-{v['broken']}" for m, v in sweep.items())]
    (ROOT / "results" / "dot_repair.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (ROOT / "results" / "dot_repair.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
