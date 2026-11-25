"""Inputs for Jev on the real candidates: crop image + our render of each candidate (NO gold text, NO labels).

    python scripts/jev/make_real_items.py      -> dist/sindhijev_jev_real.zip

Uses the EasyOCR addendum roots, which hold all engines (5 Tesseract + 2 EasyOCR) for words and lines.
Candidates are cleaned and rendered exactly as eval_real.py does for our model (Noto Naskh Arabic, size 48).
The build reads frozen input files only (crops, candidates); it is logged in docs/real_set_access_log.md.
Upload the zip as a PRIVATE Kaggle dataset only (several sources are not redistributable: docs/DATA_LICENSES.md).
"""
import csv
import datetime
import json
import sys
import zipfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sindhijev import realset as R  # noqa: E402
from sindhijev.data import clean_candidate  # noqa: E402
from sindhijev.render import Renderer  # noqa: E402

ROOTS = {"words": ROOT / "data" / "real" / "addendum_v1", "lines": ROOT / "data" / "real" / "lines" / "addendum_v1"}


def main():
    for tag, root in ROOTS.items():
        changed = R.check_frozen(root)
        assert not changed, f"{root} differs from FROZEN.json: {changed[:3]}"
    rend = Renderer(ROOT / "fonts" / "NotoNaskhArabic.ttf", size=48)
    out = ROOT / "dist" / "sindhijev_jev_real.zip"
    out.parent.mkdir(exist_ok=True)
    counts = {}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(ROOT / "scripts" / "jev" / "run_jev.py", "scripts/jev/run_jev.py")
        for tag, root in ROOTS.items():
            crops = {r["crop_id"]: r for r in R.read_csv(root / "crops.csv")}
            items, done_crops = [], set()
            for c in R.read_csv(root / "candidates.csv"):
                cid, eng = c["crop_id"], c["engine"]
                crop_rel = f"real_{tag}/crops/{cid}.png"
                if cid not in done_crops:
                    z.write(root / crops[cid]["image_path"].replace("\\", "/"), crop_rel)
                    done_crops.add(cid)
                img, _ = rend.render(clean_candidate(c["cand_text"]) or " ")
                ok, buf = cv2.imencode(".png", img)
                assert ok
                rend_rel = f"real_{tag}/renders/{cid}__{eng}.png"
                z.writestr(rend_rel, buf.tobytes())
                items.append({"id": f"{cid}|{eng}", "original_image": crop_rel.split("/", 1)[1],
                              "render_image": rend_rel.split("/", 1)[1], "ocr_text": clean_candidate(c["cand_text"])})
            z.writestr(f"real_{tag}/items.jsonl", "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items))
            counts[tag] = (len(done_crops), len(items))
    with open(ROOT / "docs" / "real_set_access_log.md", "a", encoding="utf-8") as f:
        f.write(f"- {datetime.datetime.now():%Y-%m-%d %H:%M} make_real_items (Jev inputs: crops + candidate renders, "
                f"no gold or labels) {counts}\n")
    print(f"{out}: {counts}, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
