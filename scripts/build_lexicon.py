"""Word list from the TRAIN text split only (count >= 2) -> data/text/lexicon_train.json.

Used by (1) the real-word corruption mode (data_v2) and (2) the text-only baseline's dictionary and the
non-word / real-word split of results. Never built from val/test text or the real set.

    python scripts/build_lexicon.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sindhijev.pairs import load_spans  # noqa: E402


def main(min_count=2):
    cfg = json.loads((ROOT / "configs" / "data_v1.json").read_text(encoding="utf-8"))
    wc = Counter(w for t in load_spans(ROOT / cfg["spans"], "train") for w in t.split())
    words = sorted(w for w, c in wc.items() if c >= min_count)
    out = ROOT / "data" / "text" / "lexicon_train.json"
    out.write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
    print(f"{len(words)} words (count >= {min_count}) of {len(wc)} types -> {out}")


if __name__ == "__main__":
    main()
