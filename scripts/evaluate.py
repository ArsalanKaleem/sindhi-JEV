"""Evaluate a checkpoint on the SYNTHETIC sets (never the real set; that is scripts/real/eval_real.py --final).

    python scripts/evaluate.py --ckpt runs/main/best.pt --name main [--limit 500 --device cpu]

The char threshold is chosen on synthetic val and then fixed. The text-only baseline is scored on the same
examples, and the AUROC difference gets a paired bootstrap CI. Output: results/synth_<name>.json and .md.
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
from sindhijev.baselines import TextChecker  # noqa: E402
from sindhijev.data import Manifest, collate  # noqa: E402
from sindhijev.metrics import (auroc, best_char_threshold, breakdown, paired_bootstrap_delta,  # noqa: E402
                               summarize)
from sindhijev.model import Verifier  # noqa: E402
from train import predict  # noqa: E402

SETS = ["val", "val_hard", "val_font", "val_font_hard", "test_iid", "test_font", "test_hard", "test_font_hard"]


def load_model(ckpt, dev):
    st = torch.load(ckpt, map_location=dev, weights_only=False)
    m = Verifier(st["model_cfg"]).to(dev)
    m.load_state_dict(st["model"])
    m.eval()
    return m, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    dev = torch.device(a.device)
    model, st = load_model(a.ckpt, dev)
    tc_path = ROOT / "results" / "text_checker.pkl"
    tc = TextChecker.load(tc_path) if tc_path.exists() else None
    res = {"ckpt": a.ckpt, "step": st["step"], "model_cfg": st["model_cfg"]}
    recs_by_set = {}
    for name in SETS:
        if not (ROOT / "data" / "synth" / name / "manifest.jsonl").exists():
            continue
        ds = Manifest(ROOT / "data" / "synth" / name, model.cfg, limit=a.limit)
        recs = predict(model, DataLoader(ds, batch_size=a.batch, collate_fn=collate, num_workers=2), dev,
                       dev.type == "cuda")
        for r in recs:
            r["cluster"] = r["id"]
            if tc is not None:
                r["text_score"] = tc.score(r["cand"])[0]
        recs_by_set[name] = recs
    thr = best_char_threshold(recs_by_set["val"])
    res["char_threshold_from_val"] = thr
    for name, recs in recs_by_set.items():
        out = summarize(recs, thr)
        pos = [r for r in recs if r["mode"] == "positive"]
        out["by_mode_vs_pos"] = {m: auroc([0] * len(pos) + [1] * sum(r["mode"] == m for r in recs),
                                          [r["seq_score"] for r in pos] +
                                          [r["seq_score"] for r in recs if r["mode"] == m])
                                 for m in sorted({r["mode"] for r in recs} - {"positive"})}
        neg = [r for r in recs if r["mode"] in ("confusion", "realword")]
        out["by_group_vs_pos"] = {g: summarize([r for r in neg if g in r["groups"]] + pos)["auroc"]
                                  for g in sorted({g for r in neg for g in r["groups"]})}
        out["by_font"] = breakdown(recs, lambda r: r["font"], thr)
        if tc is not None:
            out["vs_text"] = paired_bootstrap_delta(recs, "seq_score", "text_score", n=a.boot)
            out["text_auroc"] = auroc([r["seq_y"] for r in recs], [r["text_score"] for r in recs])
        res[name] = out
        print(f"{name:15s} AUROC {out['auroc']:.4f} (text {out.get('text_auroc', float('nan')):.4f}) "
              f"AUPRC {out['auprc']:.4f} charF1 {out.get('char_f1', float('nan')):.4f} "
              f"top1 {out.get('top1_loc', float('nan')):.4f}")
    od = ROOT / "results"
    od.mkdir(exist_ok=True)
    (od / f"synth_{a.name}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = [f"# Synthetic results: {a.name} (step {st['step']})", "",
             "| set | n | AUROC | text AUROC | delta [95% CI] | AUPRC | char F1 | top-1 loc | recall@10% |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name in recs_by_set:
        o = res[name]
        vt = o.get("vs_text", {})
        ci = vt.get("ci", (float("nan"), float("nan")))
        lines.append(f"| {name} | {o['n']} | {o['auroc']:.4f} | {o.get('text_auroc', float('nan')):.4f} | "
                     f"{vt.get('delta', float('nan')):+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | {o['auprc']:.4f} | "
                     f"{o.get('char_f1', float('nan')):.4f} | {o.get('top1_loc', float('nan')):.4f} | "
                     f"{o['route@10']['recall']:.4f} |")
    (od / f"synth_{a.name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
