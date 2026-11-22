"""Write data/confusion_review.csv: one row per letter pair in each group,
for the native speaker to mark. Opens in Excel (UTF-8 with BOM).

Fill in:
  is_valid_pair     Y = these really differ only as described; N = wrong pair
  ocr_confusable    H/M/L = how likely OCR confuses them in print (high/medium/low)
  dots_correct      Y/N = are both 'dots' descriptions right? (fix in comment)
  comment           free text; add missing pairs as new rows with group_id = NEW
"""
import csv
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.load(open(ROOT / "data" / "confusion_set.json", encoding="utf-8"))
    out = ROOT / "data" / "confusion_review.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "kind", "positions", "letter_a", "code_a", "dots_a",
                    "letter_b", "code_b", "dots_b", "is_valid_pair", "ocr_confusable", "dots_correct", "comment"])
        n = 0
        for g in data["groups"]:
            for a, b in itertools.combinations(g["members"], 2):
                code = lambda s: " ".join(f"U+{ord(c):04X}" for c in s)  # noqa: E731
                w.writerow([g["id"], g["kind"], g["positions"], a["c"], code(a["c"]), a["dots"],
                            b["c"], code(b["c"]), b["dots"], "", "", "", ""])
                n += 1
    print(out, n, "pairs")


if __name__ == "__main__":
    main()
