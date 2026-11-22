"""Contact sheet: every dot-confusion group and a sample sentence in each font.

    python scripts/font_sheet.py --out data/font_sheet.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sindhijev.render import Renderer  # noqa: E402

DEFAULT_FONTS = ["Amiri-Regular.ttf", "Harmattan-Regular.ttf", "Lateef-Regular.ttf", "NotoKufiArabic.ttf",
                 "NotoNaskhArabic.ttf", "NotoSansArabic.ttf", "ScheherazadeNew-Regular.ttf"]
SAMPLE = "ٻار ڀاڄي ڏٺي ڍڳو ڙيڻ ڦٽو ڇڏيو ڃڱڻ ڪتاب ۽ گهر"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts", nargs="+", default=DEFAULT_FONTS)
    ap.add_argument("--out", default=str(ROOT / "data" / "font_sheet.png"))
    args = ap.parse_args()
    groups = json.load(open(ROOT / "data" / "confusion_set.json", encoding="utf-8"))["groups"]
    line = "   ".join(" ".join(m["c"] for m in g["members"]) for g in groups if g["kind"] != "encoding")
    rows = []
    for f in args.fonts:
        r = Renderer(ROOT / "fonts" / f, size=40)
        for text in (line, SAMPLE):
            img, _ = r.render(text, pad=6)
            rows.append(img)
    W = max(i.shape[1] for i in rows)
    sheet = np.vstack([np.pad(i, ((0, 4), (W - i.shape[1], 0)), constant_values=255) for i in rows])
    Image.fromarray(sheet).save(args.out)
    print(args.out, sheet.shape)


if __name__ == "__main__":
    main()
