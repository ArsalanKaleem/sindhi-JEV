"""Phase 1: corpus -> sentence-split Sindhi text spans with a train/val/test split.

    python scripts/prepare_text.py --corpus <sd_wiki...txt> --out data/text/spans.jsonl

- normalises with sindhinorm (canonical ه, ي, ڪ; strays ZW chars removed);
- splits paragraphs into sentences at ۔ . ؟ ! ؛ and newlines;
- keeps maximal runs of words written only in Sindhi letters/diacritics
  (digits, Latin and unknown symbols break a run; we do not bidi-render them);
- assigns each SENTENCE to train/val/test by a salted hash, so every span
  cut from a sentence lands in the same split (no sentence leakage);
- drops exact duplicate sentences (the first occurrence keeps its split).
"""
import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

from sindhinorm import normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sindhijev.alphabet import DIACRITICS, EXTRA_CHARS, SINDHI_LETTERS  # noqa: E402

ALLOWED = set(SINDHI_LETTERS) | set(EXTRA_CHARS) | set(DIACRITICS)
SENT_SPLIT = re.compile(r"[۔.؟!؛?\n]+")


def split_of(sentence, salt, val=0.05, test=0.05):
    h = int(hashlib.sha1((salt + sentence).encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test else "val" if h < test + val else "train"


def clean_spans(sentence, min_words):
    spans, cur = [], []
    for w in sentence.split():
        w = w.strip("،,:\"'()[]«»-–—")
        if w and all(c in ALLOWED for c in w) and any(c in SINDHI_LETTERS for c in w):
            cur.append(w)
        else:
            if len(cur) >= min_words:
                spans.append(" ".join(cur))
            cur = []
    if len(cur) >= min_words:
        spans.append(" ".join(cur))
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default=str(ROOT / "data" / "text" / "spans.jsonl"))
    ap.add_argument("--salt", default="sindhijev-v1")
    ap.add_argument("--min-words", type=int, default=1)
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    seen, counts = set(), collections.Counter()
    with open(args.corpus, encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as out:
        for line_no, line in enumerate(f):
            for sent in SENT_SPLIT.split(normalize(line.strip())):
                sent = " ".join(sent.split())
                counts["sentences"] += 1
                if not sent or sent in seen:
                    counts["dup_or_empty"] += 1
                    continue
                seen.add(sent)
                split = split_of(sent, args.salt)
                sid = hashlib.sha1(sent.encode("utf-8")).hexdigest()[:12]
                for j, span in enumerate(clean_spans(sent, args.min_words)):
                    out.write(json.dumps({"sid": sid, "line": line_no, "span": j, "split": split,
                                          "text": span}, ensure_ascii=False) + "\n")
                    counts[f"spans_{split}"] += 1
                    counts[f"words_{split}"] += len(span.split())
    print(json.dumps(counts, indent=1))
    Path(args.out).with_suffix(".stats.json").write_text(json.dumps(
        {"corpus": args.corpus, "salt": args.salt, **counts}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
