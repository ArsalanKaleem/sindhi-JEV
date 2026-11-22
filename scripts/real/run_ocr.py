"""Phase 2: run OCR engines on every crop -> data/real/candidates.csv (Kaggle/Linux, CPU).

    apt-get install -y tesseract-ocr
    # tessdata_best models (snd urd ara fas + script/Arabic) into $TESSDATA_PREFIX
    python scripts/real/run_ocr.py --crops data/real/crops.csv --root data/real \
        --engines tess_snd tess_urd tess_ara tess_fas tess_Arabic easyocr_ur

Confidence (baseline a):
  tesseract: hOCR with hocr_char_boxes=1 -> per-char x_conf when present, word x_wconf otherwise;
             seq_conf = min over chars (or words) / 100.  Per-char parsing is checked against
             the word text; if they disagree, char confs are left empty (never guessed).
  easyocr:   per-detection confidence; seq_conf = min over detections.
Resumable: (crop_id, engine) rows already in the output are skipped.
"""
import argparse
import csv
import json
import re
import os
import subprocess
import time
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

TESS = os.environ.get("TESSERACT_CMD", "tesseract")  # binary; set to conda path on Windows
FIELDS = ["crop_id", "engine", "cand_text", "seq_conf", "char_conf_json", "runtime_s"]


def clean(s):
    return " ".join(unicodedata.normalize("NFC", s).replace("‌", "").replace("‍", "").split())


def parse_hocr(hocr):
    """hOCR (XHTML) -> (text, word_conf list, char_conf list or [])."""
    root = ET.fromstring(re.sub(r"<!DOCTYPE[^>]*>", "", hocr, count=1).encode("utf-8"))
    words, word_conf, char_conf = [], [], []
    chars_ok = True
    for el in root.iter():
        if el.get("class") != "ocrx_word":
            continue
        wtext = re.sub(r"\s+", "", "".join(el.itertext()))  # a word has no spaces; drop span-separating whitespace
        if not wtext:
            continue
        m = re.search(r"x_wconf (\d+)", el.get("title", ""))
        word_conf.append(float(m.group(1)) / 100 if m else None)
        cinfo = [c for c in el.iter() if c.get("class") == "ocrx_cinfo"]
        if cinfo and "".join((c.text or "") for c in cinfo) == wtext:
            for c in cinfo:
                mc = re.search(r"x_conf ([\d.]+)", c.get("title", ""))
                char_conf.extend([float(mc.group(1)) / 100 if mc else None] * len(c.text or ""))
            char_conf.append(None)  # the space after the word
        else:
            chars_ok = False
        words.append(wtext)
    text = clean(" ".join(words))
    cc = char_conf[:-1] if chars_ok and char_conf else []
    if cc and len(cc) != len(text):
        cc = []
    return text, word_conf, cc


PSM_FALLBACK = {7: (7, 13, 6)}  # single line -> raw line -> block (crops carry slivers of neighbour lines)


def tesseract(img, lang, psm):
    for p in PSM_FALLBACK.get(psm, (psm,)):
        out = subprocess.run([TESS, str(img), "stdout", "-l", lang, "--psm", str(p),
                              "-c", "hocr_char_boxes=1", "-c", "tessedit_create_hocr=1"],
                             capture_output=True, text=True, encoding="utf-8", check=True).stdout
        text, word_conf, cc = parse_hocr(out)
        if text:
            break
    confs = [c for c in (cc or word_conf) if c is not None]
    return text, (min(confs) if confs else None), {"chars": cc, "words": word_conf, "psm": p}


_easy = {}


def easyocr_run(img, langs):
    import easyocr  # noqa: PLC0415
    key = tuple(langs)
    if key not in _easy:
        _easy[key] = easyocr.Reader(list(langs), gpu=False)
    res = _easy[key].readtext(str(img), detail=1, paragraph=False)
    res.sort(key=lambda r: -max(p[0] for p in r[0]))  # right-to-left reading order
    text = clean(" ".join(r[1] for r in res))
    confs = [float(r[2]) for r in res]
    return text, (min(confs) if confs else None), {"detections": confs}


ENGINES = {
    "tess_snd": lambda p, psm: tesseract(p, "snd", psm),
    "tess_urd": lambda p, psm: tesseract(p, "urd", psm),
    "tess_ara": lambda p, psm: tesseract(p, "ara", psm),
    "tess_fas": lambda p, psm: tesseract(p, "fas", psm),
    "tess_Arabic": lambda p, psm: tesseract(p, "script/Arabic", psm),
    "easyocr_ur": lambda p, psm: easyocr_run(p, ["ur"]),
    "easyocr_fa": lambda p, psm: easyocr_run(p, ["fa"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="data/real/crops.csv")
    ap.add_argument("--root", default="data/real")
    ap.add_argument("--out", default="data/real/candidates.csv")
    ap.add_argument("--engines", nargs="+", default=["tess_snd", "tess_urd", "tess_ara", "tess_Arabic"])
    ap.add_argument("--psm", type=int, default=7, help="7 = single text line")
    args = ap.parse_args()
    out = Path(args.out)
    done = set()
    if out.exists():
        with open(out, encoding="utf-8") as f:
            done = {(r["crop_id"], r["engine"]) for r in csv.DictReader(f)}
    with open(args.crops, encoding="utf-8") as f:
        crops = list(csv.DictReader(f))
    new = not out.exists()
    with open(out, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        for eng in args.engines:
            for c in crops:
                if (c["crop_id"], eng) in done:
                    continue
                t0 = time.time()
                try:
                    text, conf, detail = ENGINES[eng](Path(args.root) / c["image_path"], args.psm)
                except Exception as e:  # record failures instead of dropping rows
                    text, conf, detail = "", None, {"error": repr(e)[:200]}
                w.writerow({"crop_id": c["crop_id"], "engine": eng, "cand_text": text,
                            "seq_conf": "" if conf is None else round(conf, 4),
                            "char_conf_json": json.dumps(detail, ensure_ascii=False),
                            "runtime_s": round(time.time() - t0, 3)})
                f.flush()
            print(eng, "done")


if __name__ == "__main__":
    main()
