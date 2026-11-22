"""Baseline (b), the text-only checker: train, fit on synthetic val, score synthetic sets (CPU, no GPU hours).

    python scripts/baselines.py text --max-spans 200000

Writes results/text_checker.pkl and results/synth_text_baseline.json (val / test_iid / test_font metrics,
incl. val_hard = correct vs real-word dot swaps: the decision-gate number the neural model must clearly beat).
Real-set baselines (a, b, c) are computed by scripts/real/eval_real.py --final.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sindhijev.baselines import TextChecker  # noqa: E402
from sindhijev.metrics import auroc, best_char_threshold, breakdown, summarize  # noqa: E402
from sindhijev.pairs import load_spans  # noqa: E402


def manifest(name):
    with open(ROOT / "data" / "synth" / name / "manifest.jsonl", encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def score_set(tc, rows):
    recs = []
    for r in rows:
        seq, ch = tc.score(r["cand"])
        recs.append({"seq_y": r["seq_y"] if "seq_y" in r else r["seq_label"], "seq_score": seq,
                     "char_y": r["char_labels"], "char_score": ch, "mode": r["mode"], "font": r["font"],
                     "groups": sorted({e.get("group") for e in r["edits"] if e.get("group")}), "cluster": r["id"]})
    return recs


def synth_report(recs, thr):
    out = summarize(recs, thr)
    pos = [r for r in recs if r["mode"] == "positive"]
    out["by_mode_vs_pos"] = {m: auroc([0] * len(pos) + [1] * len([r for r in recs if r["mode"] == m]),
                                      [r["seq_score"] for r in pos] +
                                      [r["seq_score"] for r in recs if r["mode"] == m])
                             for m in sorted({r["mode"] for r in recs} - {"positive"})}
    neg_by_group = [r for r in recs if r["mode"] in ("confusion", "realword")]
    out["by_group_vs_pos"] = {g: summarize([r for r in neg_by_group if g in r["groups"]] + pos, thr)["auroc"]
                              for g in sorted({g for r in neg_by_group for g in r["groups"]})}
    out["by_font"] = breakdown(recs, lambda r: r["font"], thr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["text"])
    ap.add_argument("--max-spans", type=int, default=200000)
    ap.add_argument("--order", type=int, default=5)
    a = ap.parse_args()
    cfg = json.loads((ROOT / "configs" / "data_v1.json").read_text(encoding="utf-8"))
    spans = load_spans(ROOT / cfg["spans"], "train")
    random.Random(0).shuffle(spans)
    spans = spans[:a.max_spans]
    t = time.time()
    tc = TextChecker(order=a.order).fit_lm(spans)
    lex = ROOT / "data" / "text" / "lexicon_train.json"
    if lex.exists():  # same train-only word list as the real-word corruption and the non-word/real-word split
        tc.dict = set(json.loads(lex.read_text(encoding="utf-8")))
    print(f"LM on {len(spans)} train spans in {time.time() - t:.0f}s; dictionary {len(tc.dict)} words")
    val = manifest("val")
    # fitted on BOTH synthetic val sets (v1 val + val_hard), so it also sees real-word errors: a fair baseline
    tc.fit_heads([{"cand": r["cand"], "seq_y": r["seq_label"], "char_y": r["char_labels"]}
                  for r in val + manifest("val_hard")])
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    tc.save(out_dir / "text_checker.pkl")
    val_recs = score_set(tc, val)
    thr = best_char_threshold(val_recs)
    res = {"char_threshold_from_val": thr, "train_spans": len(spans), "order": a.order}
    for name in ("val", "test_iid", "test_font", "val_hard", "test_hard", "test_font_hard", "val_font", "val_font_hard"):
        recs = val_recs if name == "val" else score_set(tc, manifest(name))
        res[name] = synth_report(recs, thr)
        print(name, {k: round(res[name][k], 4) for k in ("auroc", "auprc", "char_f1", "top1_loc")})
    (out_dir / "synth_text_baseline.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
