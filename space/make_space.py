"""Assemble a Hugging Face Space (Gradio SDK, free CPU) for the Sindhi-Jev checker.

    python space/make_space.py            -> dist/hf_space/  (upload the CONTENTS of this folder to your Space)

Contents: app.py (space/app.py), requirements.txt, README.md (Space settings), the sindhijev package (src/), the ONNX
model + model card (model/), and 24 synthetic samples in unseen fonts (data/synth/). No real test data.
Test locally first:   python dist/hf_space/app.py   -> http://127.0.0.1:7860
"""
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "hf_space"
SAMPLE_SETS = ["test_font_hard", "test_font"]
GRADIO_VERSION = "6.32.0"  # the version app.py was tested with (2026-11-26)

README = f"""---
title: Sindhi-Jev
emoji: 📝
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: {GRADIO_VERSION}
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Check Sindhi OCR text against the scan, letter by letter
---

# Sindhi-Jev checker

Checks whether Sindhi OCR text matches a word crop and flags dot errors letter by letter. 7M-parameter model, ONNX on
CPU. Code, results and limitations: https://github.com/ArsalanKaleem/sindhi-JEV

Built for crops of 1–4 printed Sindhi words; it does not work on full lines. Uploaded images are processed in memory
and not stored.
"""

REQUIREMENTS = """onnxruntime>=1.17
numpy>=1.24
opencv-python-headless>=4.8
uharfbuzz>=0.39
freetype-py>=2.4
"""


def main():
    for f in (ROOT / "dist" / "sindhijev_v2.onnx", ROOT / "dist" / "sindhijev_v2.json"):
        assert f.exists(), f"missing {f}: run scripts/calibrate.py and scripts/export_onnx.py first"
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    shutil.copy(ROOT / "space" / "app.py", OUT / "app.py")
    shutil.copytree(ROOT / "src" / "sindhijev", OUT / "src" / "sindhijev",
                    ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    (OUT / "model").mkdir()
    for f in ("sindhijev_v2.onnx", "sindhijev_v2.json"):
        shutil.copy(ROOT / "dist" / f, OUT / "model" / f)
    shutil.copy(ROOT / "LICENSE", OUT / "LICENSE")
    (OUT / "README.md").write_text(README, encoding="utf-8")
    (OUT / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    rng = random.Random(0)
    n = 0
    for s in SAMPLE_SETS:  # 6 correct + 6 wrong per set
        rows = [json.loads(x) for x in open(ROOT / "data" / "synth" / s / "manifest.jsonl", encoding="utf-8")]
        pos = [r for r in rows if r["mode"] == "positive"]
        neg = [r for r in rows if r["mode"] != "positive"]
        d = OUT / "data" / "synth" / s
        (d / "scan").mkdir(parents=True)
        with open(d / "manifest.jsonl", "w", encoding="utf-8") as f:
            for r in rng.sample(pos, 6) + rng.sample(neg, 6):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                shutil.copy(ROOT / "data" / "synth" / s / "scan" / f"{r['id']}.png", d / "scan" / f"{r['id']}.png")
                n += 1
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"{OUT}: {n} samples, {size:.1f} MB")


if __name__ == "__main__":
    main()
