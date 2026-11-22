"""Phase 2: cut word/phrase crops from real page images.

Auto mode uses Tesseract (snd model) only to FIND text lines; its text and word boxes are thrown away.
(v1 of this script used Tesseract word boxes: they split Sindhi words at non-joiner gaps and cut letters,
so nearly all crops were bad. Reviewer report 2026-11-17.)
Word boundaries come from the ink: a white column gap wider than --gap-frac x line height. Crop edges sit
halfway into those gaps, so no letter is cut. Runs of 1-4 consecutive words on one line are then sampled at
random, capped per page and per source.

    python scripts/real/make_crops.py auto --pages data/real/pages --sources data/real/sources.csv \
        --out data/real --per-page 15 --per-source 40
    # Full-line condition (separate set, same pages):
    python scripts/real/make_crops.py auto --unit line --pages data/real/pages --sources data/real/sources.csv \
        --out data/real/lines --per-page 3 --per-source 8
    # Handwriting or anything that cannot be segmented: crop by hand into a folder, then
    python scripts/real/make_crops.py manual --dir data/real/manual_crops --source-id hw_me_001 \
        --stratum handwriting --out data/real

Page images are named <source_id>__p<page>.png (or .jpg).
Known bias (report it): auto crops exist only where Tesseract found text lines and the ink has clear word gaps.
"""
import argparse
import csv
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2

TESS = os.environ.get("TESSERACT_CMD", "tesseract")  # binary; set to conda path on Windows
FIELDS = ["crop_id", "source_id", "page", "x0", "y0", "x1", "y1", "stratum", "image_path"]


def tesseract_lines(img_path, lang="snd"):
    """Text-line boxes (TSV level 4). Used ONLY to locate lines."""
    tsv = subprocess.run([TESS, str(img_path), "stdout", "-l", lang, "--psm", "3", "-c", "tessedit_create_tsv=1"],
                         capture_output=True, text=True, encoding="utf-8", check=True).stdout
    rows = list(csv.DictReader(tsv.splitlines(), delimiter="\t", quoting=csv.QUOTE_NONE))
    return [{"x0": int(r["left"]), "y0": int(r["top"]), "x1": int(r["left"]) + int(r["width"]),
             "y1": int(r["top"]) + int(r["height"])} for r in rows if r["level"] == "4"]


def ink_mask(gray):
    return cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


def line_words(ink, line, gap_frac):
    """Split a line into words at white column gaps wider than gap_frac * line height, measured on the ink.

    Returns [(x0, x1)] of complete words, left to right in image coordinates. The search extends one line
    height beyond Tesseract's box; a word running into that margin may continue, so it is dropped."""
    h = line["y1"] - line["y0"]
    X0, X1 = max(0, line["x0"] - h), min(ink.shape[1], line["x1"] + h)
    filled = ink[line["y0"]:line["y1"], X0:X1].sum(0) > 1  # tolerate 1 px of speckle per column
    runs, x, n = [], 0, len(filled)
    while x < n:
        if filled[x]:
            s = x
            while x < n and filled[x]:
                x += 1
            runs.append([s, x])
        else:
            x += 1
    words = []  # ink runs closer than the word-gap threshold belong to one word
    for s, e in runs:
        if words and s - words[-1][1] < gap_frac * h:
            words[-1][1] = e
        else:
            words.append([s, e])
    return [(X0 + a, X0 + b) for a, b in words if a > 0 and b < n]


def mean_conf(img_path, lang):
    """(mean word confidence, text) of a single-line read; used ONLY for script identity."""
    tsv = subprocess.run([TESS, str(img_path), "stdout", "-l", lang, "--psm", "7", "-c", "tessedit_create_tsv=1"],
                         capture_output=True, text=True, encoding="utf-8").stdout
    rows = [r for r in csv.DictReader(tsv.splitlines(), delimiter="\t", quoting=csv.QUOTE_NONE)
            if r.get("level") == "5" and r["text"].strip()]
    confs = [float(r["conf"]) for r in rows if float(r["conf"]) >= 0]
    return (sum(confs) / len(confs) if confs else 0.0), " ".join(r["text"] for r in rows)


def reject_reason(crop, path, latin_check):
    """Filters that do NOT use the Sindhi engine's own confidence (that would bias the test set)."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    ink = float(ink_mask(g).mean())
    if ink < 0.03:
        return "empty"
    if ink > 0.45:
        return "blob"
    m = ink_mask(g)
    if (m.mean(0) > 0.92).any() or (m.mean(1) > 0.92).any():  # page frame / column rule, not a letter
        return "rule"
    if latin_check:
        eng, etext = mean_conf(path, "eng")
        snd, _ = mean_conf(path, "snd")
        if sum(c.isascii() and c.isalpha() for c in etext) >= 3 and eng >= snd + 10:
            return "latin"
        al = [c for c in etext if c.isalnum()]
        if al and sum(c.isdigit() for c in al) >= max(2, 0.6 * len(al)):
            return "digits"
    return None


def load_sources(path):
    with open(path, encoding="utf-8-sig") as f:
        return {r["source_id"]: r for r in csv.DictReader(f)}


def append_rows(out_dir, rows):
    p = Path(out_dir) / "crops.csv"
    new = not p.exists()
    with open(p, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def crop_box(ln, span, all_words, img_shape, pad_y):
    """Box around a run of words: x edges halfway into the neighbouring white gaps (capped), never into ink."""
    h = ln["y1"] - ln["y0"]
    x0, x1 = span[0][0], span[-1][1]
    left = max((w[1] for w in all_words if w[1] <= x0), default=x0 - h)
    right = min((w[0] for w in all_words if w[0] >= x1), default=x1 + h)
    x0 = max(0, x0 - min((x0 - left) // 2, int(0.3 * h)))
    x1 = min(img_shape[1], x1 + min((right - x1) // 2, int(0.3 * h)))
    y0 = max(0, ln["y0"] - int(pad_y * h))
    y1 = min(img_shape[0], ln["y1"] + int(pad_y * h))
    return x0, y0, x1, y1


def auto(args):
    sources = load_sources(args.sources)
    per_source = defaultdict(int)
    done_pages = set()
    existing = Path(args.out) / "crops.csv"
    if existing.exists():  # resume: never re-crop a page, keep counting per-source caps
        with open(existing, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done_pages.add((r["source_id"], r["page"]))
                per_source[r["source_id"]] += 1
    crop_dir = Path(args.out) / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    rows, rejected = [], defaultdict(int)
    for page in sorted(Path(args.pages).glob("*")):
        if page.suffix.lower() not in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            continue
        sid, _, pno = page.stem.partition("__p")
        if sid not in sources:
            sys.exit(f"{page.name}: source_id {sid!r} not in sources.csv (log the license first)")
        if (sid, pno) in done_pages or sources[sid]["type"] == "excluded":
            continue
        rng = random.Random(f"{args.seed}:{page.stem}")  # per page, so resuming is reproducible
        img = cv2.imread(str(page))
        ink = ink_mask(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        lines = tesseract_lines(page)
        if not lines:
            continue
        med_h = sorted(ln["y1"] - ln["y0"] for ln in lines)[len(lines) // 2]
        spans = []  # (line index, line, words of the line, run of words)
        for li, ln in enumerate(lines):
            h = ln["y1"] - ln["y0"]
            if h < max(args.min_h, 0.6 * med_h):
                rejected["too_small"] += 1
                continue
            if h > 1.6 * med_h:
                rejected["multi_line"] += 1
                continue
            ws = line_words(ink, ln, args.gap_frac)
            if args.unit == "line":
                if len(ws) >= args.min_line_words:
                    spans.append((li, ln, ws, ws))
                continue
            for k in range(1, 5):
                for s in range(0, len(ws) - k + 1):
                    spans.append((li, ln, ws, ws[s:s + k]))
        rng.shuffle(spans)
        used, n = set(), 0
        for li, ln, ws, span in spans:
            if n >= args.per_page or per_source[sid] >= args.per_source:
                break
            key = {(li, w) for w in span}
            if key & used:  # no overlapping crops
                continue
            x0, y0, x1, y1 = crop_box(ln, span, ws, img.shape, args.pad_y)
            if x1 - x0 < 0.6 * (ln["y1"] - ln["y0"]):
                rejected["too_small"] += 1
                continue
            if args.unit == "words" and x1 - x0 > args.max_word_w * (ln["y1"] - ln["y0"]):
                rejected["no_word_gaps"] += 1  # gaps too tight to find words: a 1-4 word span would be a line
                continue
            used |= key
            cid = f"{sid}_p{pno}_v2{'L' if args.unit == 'line' else 'w'}{n:02d}"  # v2: never reuse v1 ids (browser-saved labels)
            path = crop_dir / f"{cid}.png"
            crop = img[y0:y1, x0:x1]
            cv2.imwrite(str(path), crop)
            why = reject_reason(crop, path, args.latin_check)
            if why:
                rejected[why] += 1
                path.unlink()
                continue
            rows.append({"crop_id": cid, "source_id": sid, "page": pno, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                         "stratum": sources[sid]["type"], "image_path": str(path.relative_to(args.out))})
            n += 1
            per_source[sid] += 1
    append_rows(args.out, rows)
    print(f"{len(rows)} new crops; rejected: {dict(rejected)}")
    log = Path(args.out) / "crop_rejections.json"
    old = json.loads(log.read_text(encoding="utf-8")) if log.exists() else {}
    for k, v in rejected.items():
        old[k] = old.get(k, 0) + v
    log.write_text(json.dumps(old, indent=1), encoding="utf-8")


def manual(args):
    crop_dir = Path(args.out) / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, f in enumerate(sorted(Path(args.dir).glob("*"))):
        if f.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        img = cv2.imread(str(f))
        cid = f"{args.source_id}_m{i:03d}"
        path = crop_dir / f"{cid}.png"
        cv2.imwrite(str(path), img)
        rows.append({"crop_id": cid, "source_id": args.source_id, "page": "", "x0": "", "y0": "", "x1": "",
                     "y1": "", "stratum": args.stratum, "image_path": str(path.relative_to(args.out))})
    append_rows(args.out, rows)
    print(f"{len(rows)} manual crops")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("auto")
    a.add_argument("--pages", required=True)
    a.add_argument("--sources", required=True)
    a.add_argument("--out", default="data/real")
    a.add_argument("--per-page", type=int, default=15)
    a.add_argument("--per-source", type=int, default=40)
    a.add_argument("--seed", type=int, default=7)
    a.add_argument("--unit", choices=["words", "line"], default="words",
                   help="words = runs of 1-4 words; line = whole lines (use a separate --out, e.g. data/real/lines)")
    a.add_argument("--min-line-words", type=int, default=5)
    a.add_argument("--gap-frac", type=float, default=0.15,
                   help="white gap wider than this x line height = word boundary (narrower = inside a word)")
    a.add_argument("--pad-y", type=float, default=0.1, help="vertical padding, x line height")
    a.add_argument("--min-h", type=int, default=20, help="minimum line height in px (drops tiny print)")
    a.add_argument("--max-word-w", type=float, default=10, help="max width of a 1-4 word crop, x line height")
    a.add_argument("--no-latin-check", dest="latin_check", action="store_false",
                   help="skip the eng-vs-snd script and digits check (needs eng.traineddata)")
    m = sub.add_parser("manual")
    m.add_argument("--dir", required=True)
    m.add_argument("--source-id", required=True)
    m.add_argument("--stratum", default="handwriting")
    m.add_argument("--out", default="data/real")
    args = ap.parse_args()
    auto(args) if args.cmd == "auto" else manual(args)


if __name__ == "__main__":
    main()
