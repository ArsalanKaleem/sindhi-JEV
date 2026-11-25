"""Assemble a Hugging Face Space (Docker) that serves the docs site and the checker API.

    python site_src/build.py
    python space/make_space.py            -> dist/hf_space/  (upload this folder to your Space)

Contents: the sindhijev package, tools/demo/server.py, the built site, the ONNX model + model card, and 24 synthetic
samples (unseen fonts). No real test data. Test it locally before uploading:

    python dist/hf_space/tools/demo/server.py --port 8780 --weights dist/hf_space/model/sindhijev_v2.onnx
"""
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "hf_space"
SAMPLE_SETS = ["test_font_hard", "test_font"]

README = """---
title: Sindhi-Jev
emoji: 📝
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
license: apache-2.0
short_description: Check Sindhi OCR text against the scan, letter by letter
---

# Sindhi-Jev checker

Docs and playground for [Sindhi-Jev](https://github.com/ArsalanKaleem/sindhi-JEV): a 7M-parameter model that checks
Sindhi OCR text against a word crop and flags dot errors letter by letter. Runs on CPU (ONNX).

Built for word crops of printed Naskh-style text. Uploaded images are processed in memory and not stored.
"""

DOCKERFILE = """FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY . /app
RUN pip install /app
RUN useradd -m -u 1000 user && chown -R user /app
USER user
EXPOSE 7860
CMD ["python", "tools/demo/server.py", "--host", "0.0.0.0", "--port", "7860", "--weights", "model/sindhijev_v2.onnx"]
"""


def main():
    for f in (ROOT / "dist" / "sindhijev_v2.onnx", ROOT / "dist" / "sindhijev_v2.json", ROOT / "site" / "index.html"):
        assert f.exists(), f"missing {f}: run scripts/export_onnx.py and site_src/build.py first"
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    shutil.copytree(ROOT / "src" / "sindhijev", OUT / "src" / "sindhijev",
                    ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    shutil.copytree(ROOT / "site", OUT / "site")
    (OUT / "tools" / "demo").mkdir(parents=True)
    shutil.copy(ROOT / "tools" / "demo" / "server.py", OUT / "tools" / "demo" / "server.py")
    (OUT / "model").mkdir()
    for f in ("sindhijev_v2.onnx", "sindhijev_v2.json"):
        shutil.copy(ROOT / "dist" / f, OUT / "model" / f)
    shutil.copy(ROOT / "pyproject.toml", OUT / "pyproject.toml")
    shutil.copy(ROOT / "LICENSE", OUT / "LICENSE")
    (OUT / "README.md").write_text(README, encoding="utf-8")
    (OUT / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8")
    (OUT / ".gitattributes").write_text("*.onnx filter=lfs diff=lfs merge=lfs -text\n*.ttf filter=lfs diff=lfs merge=lfs -text\n",
                                        encoding="utf-8")
    rng = random.Random(0)
    n = 0
    for s in SAMPLE_SETS:  # same choice rule as server.load_samples: 6 correct + 6 wrong per set
        rows = [json.loads(x) for x in open(ROOT / "data" / "synth" / s / "manifest.jsonl", encoding="utf-8")]
        pos = [r for r in rows if r["mode"] == "positive"]
        neg = [r for r in rows if r["mode"] != "positive"]
        pick = rng.sample(pos, 6) + rng.sample(neg, 6)
        d = OUT / "data" / "synth" / s
        (d / "scan").mkdir(parents=True)
        with open(d / "manifest.jsonl", "w", encoding="utf-8") as f:
            for r in pick:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                shutil.copy(ROOT / "data" / "synth" / s / "scan" / f"{r['id']}.png", d / "scan" / f"{r['id']}.png")
                n += 1
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"{OUT}: {n} samples, {size:.1f} MB")


if __name__ == "__main__":
    main()
