# Sindhi-Jev: a small dot-aware verifier for Sindhi OCR

[![tests](https://github.com/ArsalanKaleem/sindhi-JEV/actions/workflows/tests.yml/badge.svg)](https://github.com/ArsalanKaleem/sindhi-JEV/actions/workflows/tests.yml)

Checks whether an OCR candidate for a Sindhi word/line crop is correct, and
which character's dots disagree. See `docs/phase0_audit.md` for prior work
(MiniCPM-OCR Jev, OCR-EDR) and the claim being tested.

## Status

Design decisions: `docs/decisions.md`.

| Phase | State | GPU h |
|---|---|---|
| 0 Novelty audit | approved 2026-11-17 | 0 |
| 1 Data (synthetic) | built with confusion set **v1** (native-reviewed 2026-11-17) | 0 |
| 2 Real test set | **frozen**: 336 word crops + 74 lines labelled; 5 Tesseract + 2 EasyOCR engines (2,352 + 530 outputs) | 0 |
| 3 Model v1 | trained 60k steps; gate passed (val_hard +0.151 over text-only, CI above 0); weak on unseen fonts | ~8.2 |
| 3 Model v2 | 9 fonts + page realism; unseen-font val 0.9997 vs v1 0.9834 at step 51k; finishing to 60k ([#1](https://github.com/ArsalanKaleem/sindhi-JEV/issues/1)) | ~8.4 |
| Next | ablations, Jev baseline, final real evaluation, paper, pip tool: see the [Roadmap](https://github.com/ArsalanKaleem/sindhi-JEV/issues/14) | ~16.6 / 60 used |

## Setup (Windows or Linux, Python 3.12)

```bash
py -3.12 -m venv .venv            # Linux: python3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.lock.txt
.venv/Scripts/python -m pip install ../Morph/morphtok-sd-ur/packages/sindhinorm   # or: pip install sindhinorm
```

Rendering uses HarfBuzz (`uharfbuzz`) + FreeType directly, so it does **not**
need Pillow's libraqm and gives the same output on Windows and Kaggle.
On Windows set `PYTHONUTF8=1` so Sindhi prints to the console.

## Phase 1 pipeline

```bash
python scripts/check_fonts.py                      # -> data/font_coverage.md
python scripts/font_sheet.py                       # -> data/font_sheet.png (eyeball dots)
python scripts/prepare_text.py --corpus <sd_wiki.txt>   # -> data/text/spans.jsonl
python scripts/build_pairs.py --sets val test_iid test_font   # -> data/synth/<set>/
python scripts/pair_sheet.py data/synth/val --only-neg --out data/synth/val_neg_sheet.png
python scripts/export_confusion_review.py          # -> data/confusion_review.csv
python -m pytest tests
```

| Piece | File |
|---|---|
| Alphabet, digraph units, code-point checks | `src/sindhijev/alphabet.py` |
| Dot-confusion set (v1, reviewed; changes in `docs/confusion_set_changelog.md`) | `data/confusion_set.json` |
| Renderer (HarfBuzz/FreeType, per-char x-ranges) | `src/sindhijev/render.py` |
| Scan degradation | `src/sindhijev/degrade.py` |
| Candidate corruption + per-char labels | `src/sindhijev/corrupt.py` |
| Pair generator (fixed sets and on-the-fly training) | `src/sindhijev/pairs.py` |
| Data config (font split, mix, sizes) | `configs/data_v1.json` |

### Splits (no leakage)

- **By sentence:** each normalised sentence is hashed (salt `sindhijev-v1`) into
  train 90% / val 5% / test 5%; all spans of a sentence share its split;
  duplicate sentences are dropped.
- **By font:** scans in `test_font` use fonts never used for training scans
  (Harmattan, Noto Kufi Arabic). Candidates are always rendered in Noto Naskh
  Arabic, as a deployed verifier would (it never knows the scan font).

### Label rules

Labels are per code point of the candidate. Swap/substitution: the new unit's
code points = 1. Insertion: the inserted code point = 1. Deletion: the code
point right after the gap = 1 (before it, if at the end). If a corrupted
candidate renders pixel-identical to the truth in the scan font, it is relabelled
positive (`mode = invisible`); 0 such cases in the current sets.

## Phase 2 tools (real test set)

Protocol: `docs/labeling_protocol.md`. Source leads: `docs/real_sources_candidates.md`.

```bash
# Kaggle CPU notebook (0 GPU h): crops + OCR candidates
bash kaggle/phase2_ocr_engines.sh /kaggle/input/sindhi-jev
# Local: label (server mode; or open tools/labeler.html directly and pick the files)
python -m http.server 8765    # then http://localhost:8765/tools/labeler.html?data=../data/real
python scripts/real/realset_cli.py derive          # gold.csv -> candidate_labels.csv
python scripts/real/realset_cli.py freeze          # lock the set
python scripts/real/realset_cli.py relabel-sample  # one week later: 10% blind queue
#   label with ?data=../data/real&pass=2, export, then:
python scripts/real/realset_cli.py agreement --pass2 data/real/gold_pass2_<date>.csv
```

Practice set (synthetic scans, not real): `?data=../data/demo_real`.

## Known limitations (Phase 1)

- Confusion set v1 was reviewed by one native speaker (the author); no second reviewer.
- Degradation could in principle erase a dot and make a "correct" candidate
  wrong; severity is capped at 0.8 and erosion is limited, but this is not yet
  measured.
- Rare groups have few test negatives (e.g. tah, sad, gaf_digraph < 10 per set);
  a per-group balanced set is needed before reporting per-group numbers.
- Text is Wikipedia only (encyclopedic register, 2023 dump).

## Licence

- **Code** (src/, scripts/, tools/, kaggle/, tests/): [Apache License 2.0](LICENSE).
- **Fonts** (fonts/): each under the SIL Open Font License; licence texts are next to the fonts (`fonts/OFL-*.txt`).
- **Data**: see [docs/DATA_LICENSES.md](docs/DATA_LICENSES.md). Real scan images are not in this repository;
  each real source's terms are in `data/real/sources.csv`.
