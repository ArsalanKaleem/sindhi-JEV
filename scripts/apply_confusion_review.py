"""Fold the native-speaker review (data/confusion_review.csv) into data/confusion_set.json (draft-1 -> v1).

Each group gets an explicit "pairs" list (only pairs marked is_valid_pair=Y), each with the reviewer's
H/M/L OCR-confusability level. Invalid pairs go to "excluded_pairs" with the reviewer's comment.
Dot descriptions marked dots_correct=N are replaced from DOT_FIXES (written from the reviewer's comments).
The draft is kept as data/confusion_set_draft1.json. Changes are logged to docs/confusion_set_changelog.md.

    python -I scripts/apply_confusion_review.py
"""
import csv
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEVEL_MULT = {"h": 1.0, "m": 0.5, "l": 0.2}  # sampling multiplier per pair; documented in the JSON notes
# reviewer comment -> corrected description (dots_correct = N rows)
DOT_FIXES = {
    "ی": "Farsi/Urdu yeh: no dots in final/isolated form; Urdu/Persian code point, rare in Sindhi text (reviewer)",
    "ك": "Arabic kaf; Urdu/Arabic code point, rarely used in Sindhi text (reviewer)",
}


def main():
    src = ROOT / "data" / "confusion_set.json"
    data = json.loads(src.read_text(encoding="utf-8"))
    if data["version"] != "draft-1":
        raise SystemExit(f"expected draft-1, found {data['version']} (already applied?)")
    (ROOT / "data" / "confusion_set_draft1.json").write_text(json.dumps(data, ensure_ascii=False, indent=1),
                                                             encoding="utf-8")
    with open(ROOT / "data" / "confusion_review.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if any(r["group_id"] == "NEW" for r in rows):
        raise SystemExit("NEW rows present: add them by hand first")
    by_group = defaultdict(list)
    for r in rows:
        by_group[r["group_id"]].append(r)
    log = []
    for g in data["groups"]:
        pairs, excl = [], []
        for r in by_group[g["id"]]:
            a, b = r["letter_a"], r["letter_b"]
            valid = r["is_valid_pair"].strip().lower() == "y"
            lvl = r["ocr_confusable"].strip().upper()
            assert lvl in ("H", "M", "L"), r
            (pairs if valid else excl).append({"a": a, "b": b, "level": lvl, **({"comment": r["comment"].strip()}
                                                                              if r["comment"].strip() else {})})
            if not valid:
                log.append(f"- `{g['id']}`: pair {a} / {b} marked **invalid** ({lvl})"
                           + (f": {r['comment'].strip()}" if r["comment"].strip() else ""))
            if r["dots_correct"].strip().lower() == "n":
                for m in g["members"]:
                    if m["c"] in DOT_FIXES and m["c"] in (a, b) and m["dots"] != DOT_FIXES[m["c"]]:
                        log.append(f"- `{g['id']}`: description of {m['c']} changed from \"{m['dots']}\" "
                                   f"to \"{DOT_FIXES[m['c']]}\"")
                        m["dots"] = DOT_FIXES[m["c"]]
        g["pairs"] = pairs
        if excl:
            g["excluded_pairs"] = excl
        if not pairs and g["kind"] != "encoding":
            g["weight"] = 0.0
            log.append(f"- `{g['id']}`: no valid pairs left, weight set to 0")
    data["version"] = "v1"
    data["status"] = f"Native-speaker reviewed {date.today().isoformat()} (data/confusion_review.csv)."
    data["notes"] += [
        "pairs: the only swaps used for synthetic negatives and for tagging real errors with a group. "
        f"Sampling weight of a swap = group weight x level multiplier {LEVEL_MULT} / number of partners.",
        "excluded_pairs: rejected by the reviewer; never generated, tagged 'other' in real-error analysis.",
        "kind=encoding groups keep their invalid pairs in tagging (they are code-point variants, reported "
        "separately and monitored, not counted in the main OCR error metric; decision 2026-11-17).",
    ]
    data["level_mult"] = LEVEL_MULT
    src.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    cl = ROOT / "docs" / "confusion_set_changelog.md"
    cl.write_text(f"# Confusion set changelog\n\n## v1 ({date.today().isoformat()}): native-speaker review\n\n"
                  f"{len(rows)} pairs reviewed; {sum(len(g['pairs']) for g in data['groups'])} valid, "
                  f"{sum(len(g.get('excluded_pairs', [])) for g in data['groups'])} excluded.\n\n"
                  + "\n".join(log) + "\n\nDraft kept as `data/confusion_set_draft1.json`.\n", encoding="utf-8")
    print("\n".join(log))


if __name__ == "__main__":
    main()
