"""Phase 1: write the FIXED synthetic evaluation sets (and optional train shards).

    python scripts/build_pairs.py --config configs/data_v1.json --sets val test_iid test_font
    python scripts/build_pairs.py --config configs/data_v1.json --sets train --n 2000   # debug shard

Sets:
  val       : val sentences,  train scan fonts
  test_iid  : test sentences, train scan fonts
  test_font : test sentences, HELD-OUT scan fonts
  train     : train sentences, train scan fonts (normally generated on the fly)

Output: data/synth/<set>/{scan,cand}/<id>.png and data/synth/<set>/manifest.jsonl
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sindhijev.pairs import PairGenerator, example_rng  # noqa: E402

SET_SPEC = {"val": ("val", "scan_fonts_train"), "test_iid": ("test", "scan_fonts_train"),
            "test_font": ("test", "scan_fonts_heldout"), "train": ("train", "scan_fonts_train"),
            # hard sets (data_v2, need cfg["lexicon"]): 50% correct, 50% REAL-WORD dot swaps, which a
            # dictionary cannot flag; the decision gate and the "image is needed" claim are measured here
            "val_hard": ("val", "scan_fonts_train"), "test_hard": ("test", "scan_fonts_train"),
            "test_font_hard": ("test", "scan_fonts_heldout"),
            # v3: unseen-font VALIDATION (Mirza), to compare models without touching the test fonts
            "val_font": ("val", "scan_fonts_val"), "val_font_hard": ("val", "scan_fonts_val")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "data_v1.json"))
    ap.add_argument("--sets", nargs="+", default=["val", "test_iid", "test_font"])
    ap.add_argument("--n", type=int, default=None, help="override set size")
    ap.add_argument("--out", default=str(ROOT / "data" / "synth"))
    args = ap.parse_args()
    cfg = json.load(open(args.config, encoding="utf-8"))
    for name in args.sets:
        split, font_key = SET_SPEC[name]
        n = args.n or cfg["fixed_sets"].get(name, 2000)
        gen = PairGenerator(cfg, split, cfg[font_key])
        d = Path(args.out) / name
        (d / "scan").mkdir(parents=True, exist_ok=True)
        (d / "cand").mkdir(parents=True, exist_ok=True)
        counts = collections.Counter()
        with open(d / "manifest.jsonl", "w", encoding="utf-8") as man:
            for i in range(n):
                mode = ("positive" if i % 2 == 0 else "realword") if name.endswith("_hard") else None
                if mode == "realword" and not gen.lexicon:
                    sys.exit("hard sets need a lexicon: use configs/data_v2.json")
                ex = gen.sample(example_rng(cfg["seed"], name, i), mode=mode)
                eid = f"{name}_{i:06d}"
                cv2.imwrite(str(d / "scan" / f"{eid}.png"), ex.pop("scan"))
                cv2.imwrite(str(d / "cand" / f"{eid}.png"), ex.pop("cand_img"))
                ex["id"] = eid
                man.write(json.dumps(ex, ensure_ascii=False) + "\n")
                counts[ex["mode"]] += 1
                counts[f"font:{ex['font']}"] += 1
                for e in ex["edits"]:
                    if e["op"] == "swap":
                        counts[f"group:{e['group']}"] += 1
        stats = {"set": name, "n": n, "relabelled_invisible": gen.relabelled, **dict(sorted(counts.items()))}
        (d / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
