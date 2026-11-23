"""Zip what the Jev zero-shot notebook needs: the client and two synthetic test sets (no real data).

    python kaggle/make_jev_bundle.py      -> dist/sindhijev_jev.zip (~60 MB)

Upload as a private Kaggle dataset "sindhijev-jev"; then run kaggle/jev.ipynb on a T4 with Internet ON.
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["scripts/jev/run_jev.py", "data/synth/test_hard", "data/synth/test_font_hard"]


def main():
    out = ROOT / "dist" / "sindhijev_jev.zip"
    out.parent.mkdir(exist_ok=True)
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for inc in INCLUDE:
            p = ROOT / inc
            for f in [p] if p.is_file() else sorted(q for q in p.rglob("*") if q.is_file()):
                z.write(f, f.relative_to(ROOT).as_posix())
                n += 1
    print(f"{out}: {n} files, {out.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
