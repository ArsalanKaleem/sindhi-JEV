"""Frozen ENGINE ADDENDUM: the frozen real set + outputs of extra OCR engines (approved 2026-11-20).

    python scripts/real/build_addendum.py --new data/real/addendum_v1/candidates_new.csv --out data/real/addendum_v1

1. Checks the original frozen set is unchanged (data/real/FROZEN.json).
2. Copies crops.csv, gold.csv, sources.csv and crop images unchanged; candidates.csv = frozen candidates + new rows.
3. Derives labels from the SAME frozen gold (no new labelling; pass-1 overrides ignored, as for the main set)
   and freezes the addendum with its own FROZEN.json.
The original frozen files are never modified. Engines run on crops only; nobody looks at gold or errors here.
"""
import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sindhijev import realset as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data" / "real"))
    ap.add_argument("--new", required=True, nargs="+", help="candidate CSVs from run_ocr.py with extra engines")
    ap.add_argument("--out", default=str(ROOT / "data" / "real" / "addendum_v1"))
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    changed = R.check_frozen(src)
    if changed:
        sys.exit(f"original frozen set changed: {changed[:5]}")
    if (out / "FROZEN.json").exists():
        sys.exit(f"{out} is already frozen; make a new addendum version instead")
    old = R.read_csv(src / "candidates.csv")
    crop_ids = {r["crop_id"] for r in R.read_csv(src / "crops.csv")}
    old_engines = {r["engine"] for r in old}
    new = []
    for p in a.new:
        new += R.read_csv(p)
    clash = {r["engine"] for r in new} & old_engines
    if clash:
        sys.exit(f"engines already in the frozen set: {clash}")
    for eng in {r["engine"] for r in new}:
        have = {r["crop_id"] for r in new if r["engine"] == eng}
        if have != crop_ids:
            sys.exit(f"{eng}: covers {len(have)} of {len(crop_ids)} crops")
    out.mkdir(parents=True, exist_ok=True)
    for n in ("crops.csv", "gold.csv", "sources.csv"):
        shutil.copy2(src / n, out / n)
    if (out / "crops").exists():
        shutil.rmtree(out / "crops")
    shutil.copytree(src / "crops", out / "crops")
    with open(out / "candidates.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, list(old[0].keys()))
        w.writeheader()
        w.writerows(old + [{k: r.get(k, "") for k in old[0]} for r in new])
    cli = [sys.executable, str(ROOT / "scripts" / "real" / "realset_cli.py")]
    subprocess.run(cli + ["derive", "--root", str(out), "--ignore-overrides"], check=True)
    subprocess.run(cli + ["freeze", "--root", str(out)], check=True)
    print(f"addendum: {len(old)} frozen + {len(new)} new candidate rows -> {out}")


if __name__ == "__main__":
    main()
