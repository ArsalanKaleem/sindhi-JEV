"""Contact sheet of built pairs: scan (left) | candidate render (right), with
mismatched candidate characters shaded red using cand_char_x.

    python scripts/pair_sheet.py data/synth/val --n 24 --out data/synth/val_sheet.png
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set_dir")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--only-neg", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = Path(args.set_dir)
    rows, H = [], 56
    for line in open(d / "manifest.jsonl", encoding="utf-8"):
        ex = json.loads(line)
        if args.only_neg and not ex["seq_label"]:
            continue
        scan = cv2.imread(str(d / "scan" / f"{ex['id']}.png"), cv2.IMREAD_GRAYSCALE)
        cand = cv2.cvtColor(cv2.imread(str(d / "cand" / f"{ex['id']}.png"), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
        for (x0, x1), lab in zip([x or (0, 0) for x in ex["cand_char_x"]], ex["char_labels"]):
            if lab:
                cand[:, x0:x1, :2] = (cand[:, x0:x1, :2] * 0.55).astype(np.uint8)  # BGR: keep red
        scan = cv2.cvtColor(cv2.resize(scan, (max(1, scan.shape[1] * H // scan.shape[0]), H)), cv2.COLOR_GRAY2BGR)
        cand = cv2.resize(cand, (max(1, cand.shape[1] * H // cand.shape[0]), H))
        tag = np.full((H, 8, 3), (0, 0, 220) if ex["seq_label"] else (0, 180, 0), np.uint8)
        rows.append(np.hstack([tag, scan, np.full((H, 20, 3), 200, np.uint8), cand]))
        if len(rows) >= args.n:
            break
    W = max(r.shape[1] for r in rows)
    sheet = np.vstack([np.pad(r, ((0, 4), (0, W - r.shape[1]), (0, 0)), constant_values=255) for r in rows])
    cv2.imwrite(args.out, sheet)
    print(args.out, sheet.shape)


if __name__ == "__main__":
    main()
