"""Phase 2: inspect downloaded PDFs and render a seeded random sample of pages.

    python -I scripts/real/pdf_pages.py inspect --pdfs data/real/raw_pdfs
    python -I scripts/real/pdf_pages.py sample  --pdfs data/real/raw_pdfs --out data/real/pages --n 6

inspect: pages, share of pages with a text layer, share of pages that are one big image
         (=> scan) vs vector text only (=> born-digital / retyped).
sample:  N pages per PDF, skipping the first 3 and last 2 (covers, contents, index),
         rendered so the long side is ~2400 px, saved grayscale as <source_id>__p<page>.png.
PDFs are treated as untrusted data: only rendered with pdfium, nothing is executed.
"""
import argparse
import json
import random
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as R


def page_kind(page):
    W, H = page.get_size()
    img_area, n_img = 0.0, 0
    for obj in page.get_objects(filter=[R.FPDF_PAGEOBJ_IMAGE]):
        l, b, r, t = obj.get_bounds()
        img_area += max(0, r - l) * max(0, t - b)
        n_img += 1
    tp = page.get_textpage()
    nchar = tp.count_chars()
    tp.close()
    cover = img_area / (W * H) if W * H else 0
    return {"image_cover": round(cover, 2), "n_images": n_img, "text_chars": nchar}


ONLY = None  # --only: explicit source_ids (use while a download is still running)


def finished_pdfs(pdf_dir):
    """PDFs whose size matches download_log.json (skips files still downloading)."""
    if ONLY:
        return [p for p in sorted(Path(pdf_dir).glob("*.pdf")) if p.stem in ONLY]
    log = Path(pdf_dir) / "download_log.json"
    ok = {x["source_id"] for x in json.loads(log.read_text(encoding="utf-8")) if x["ok"]} if log.exists() else None
    return [p for p in sorted(Path(pdf_dir).glob("*.pdf")) if ok is None or p.stem in ok]


def inspect(pdf_dir):
    rows = []
    for p in finished_pdfs(pdf_dir):
        doc = pdfium.PdfDocument(str(p))
        n = len(doc)
        idx = sorted(random.Random(0).sample(range(n), min(n, 12)))
        kinds = [page_kind(doc[i]) for i in idx]
        scan = sum(k["image_cover"] > 0.6 for k in kinds) / len(kinds)
        text = sum(k["text_chars"] > 50 for k in kinds) / len(kinds)
        verdict = "scan" if scan >= 0.8 else "born-digital" if scan <= 0.2 and text >= 0.5 else "mixed"
        rows.append({"source_id": p.stem, "pages": n, "scan_share": round(scan, 2), "text_layer_share": round(text, 2),
                     "verdict": verdict})
        print(f"{p.stem:20s} pages={n:4d} scan={scan:.2f} text={text:.2f} -> {verdict}")
        doc.close()
    (Path(pdf_dir) / "inspect.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


def sample(pdf_dir, out, n, seed, long_side):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for p in finished_pdfs(pdf_dir):
        if list(out.glob(f"{p.stem}__p*.png")):
            continue  # already sampled
        doc = pdfium.PdfDocument(str(p))
        N = len(doc)
        pool = list(range(3, N - 2)) if N > 8 else list(range(N))
        pick = sorted(random.Random(f"{seed}:{p.stem}").sample(pool, min(n, len(pool))))
        for i in pick:
            page = doc[i]
            W, H = page.get_size()
            scale = long_side / max(W, H)
            img = page.render(scale=scale, grayscale=True).to_pil()
            img.save(out / f"{p.stem}__p{i + 1:04d}.png")
        print(p.stem, "pages", [i + 1 for i in pick])
        doc.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["inspect", "sample"])
    ap.add_argument("--pdfs", default="data/real/raw_pdfs")
    ap.add_argument("--out", default="data/real/pages")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--long-side", type=int, default=2400)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    global ONLY
    ONLY = set(a.only) if a.only else None
    inspect(a.pdfs) if a.cmd == "inspect" else sample(a.pdfs, a.out, a.n, a.seed, a.long_side)


if __name__ == "__main__":
    main()
