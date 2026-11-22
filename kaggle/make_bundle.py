"""Zip what a Kaggle GPU session needs to TRAIN (evaluation runs locally on CPU: no GPU hours).

    python kaggle/make_bundle.py            -> dist/sindhijev_kaggle.zip (~150 MB)

Upload it once as a private Kaggle dataset named "sindhijev-code" (Datasets -> New -> upload the zip;
Kaggle unzips it). Re-upload a new version only when code or data change.
Contents: src/ scripts/ configs/ fonts/ data/confusion_set.json data/text/{spans.jsonl,lexicon_train.json}
data/synth/{val,val_hard}/. The real set is NEVER included.
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["src/sindhijev", "scripts/train.py", "scripts/make_shards.py", "scripts/evaluate.py", "configs", "fonts",
           "data/confusion_set.json", "data/text/spans.jsonl", "data/text/lexicon_train.json",
           "data/synth/val", "data/synth/val_hard", "data/synth/val_font_hard", "requirements.lock.txt"]
SKIP = {"__pycache__"}


def main():
    out = ROOT / "dist" / "sindhijev_kaggle.zip"
    out.parent.mkdir(exist_ok=True)
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for inc in INCLUDE:
            p = ROOT / inc
            files = [p] if p.is_file() else sorted(q for q in p.rglob("*") if q.is_file())
            for f in files:
                rel = f.relative_to(ROOT)
                if SKIP & set(rel.parts) or "real" in rel.parts:
                    continue
                z.write(f, rel.as_posix())
                n += 1
    print(f"{out}: {n} files, {out.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
