# Phase 2: real test set (labelling protocol)

Goal: 300–500 real Sindhi crops, each with a gold transcription. Every OCR
engine's output on every crop then gets a sequence label (correct/incorrect)
and per-character labels **derived automatically** from the gold text
(`src/sindhijev/align.py`, same rules as the synthetic data). You verify the
derived marks in the labeller and can override any mark.

Why gold text instead of marking each OCR output by hand: one transcription
labels all engines consistently (4+ engines × 400 crops = 1,600+ outputs), and
the same gold works for any engine added later (e.g. the VLM baseline).

**The set is UNTOUCHED until final evaluation.** No model selection, threshold
tuning or error browsing on it. Thresholds come from the synthetic `val` set.
`scripts/real/freeze.py` writes hashes; `eval` code refuses to load it without
`--final`.

## 1. Target composition

| Stratum | Target crops | Source type | License handling |
|---|---|---|---|
| Printed books / periodicals | 200–250 | public-domain or openly licensed scans (archive.org, Wikimedia Commons, publisher-permitted PDFs) | redistribute crops if PD/CC; log each item |
| Newspapers | 100–150 | e-paper images / scans | usually copyrighted: research eval only; release **only** source URL + bbox + gold text if terms allow, never images |
| Handwriting | 50–100 | **your own** handwriting (and volunteers with written consent) | you own it; release under CC BY 4.0 |

Handwriting text: write **new sentences or text from outside Wikipedia**, never
sentences from `data/text/spans.jsonl` (they may be in training).

Crop unit: one **word or short phrase (1–4 words)** to match the synthetic data;
plus a separate set of **full lines** (`data/real/lines/`, decided 2026-11-17, see `docs/decisions.md`),
labelled and reported separately so results can be compared by crop length.

Sampling: take crops at random positions on each page (script picks them), not
the "interesting" ones; cap 15 crops per page and 40 per source so one book
does not dominate. Slivers of the neighbouring lines at the top/bottom edge are normal: ignore them.

## 2. Files (all in `data/real/`, UTF-8)

`sources.csv`: one row per page/image.
```
source_id,url_or_origin,title,type(printed|newspaper|handwriting),year,license,may_redistribute(Y/N),retrieved,notes
```

`crops.csv`: one row per crop (made by `scripts/real/make_crops.py` or by hand).
```
crop_id,source_id,page,x0,y0,x1,y1,stratum,image_path
```

`candidates.csv`: one row per (crop, engine) (made by `scripts/real/run_ocr.py`).
```
crop_id,engine,cand_text,seq_conf,char_conf_json,runtime_s
```
`seq_conf` = engine's own confidence in [0,1] (baseline a); `char_conf_json` =
per-code-point confidences if the engine gives them, else empty.

`gold.csv`: written by the labeller, one row per crop and labelling pass.
```
crop_id,pass(1|2),gold_text,status(ok|illegible|not_sindhi|bad_crop),ambiguous_chars,override_json,seconds,labeled_at
```
- `gold_text`: exactly what is printed, in canonical Sindhi code points (ه, ي, ڪ),
  diacritics only if printed.
- `ambiguous_chars`: positions in gold_text where the **scan itself** cannot
  decide the dots (e.g. smudged). These are excluded from character metrics.
- `override_json`: your corrections to the auto-derived marks, `{engine: [positions]}`.

`candidate_labels.csv`: derived by `scripts/real/derive_labels.py`.
```
crop_id,engine,seq_label,char_labels_json,errors_json,overridden(Y/N)
```

## 3. Labelling rules

1. Read the crop, not the OCR. The textbox is pre-filled with one engine's output
   only to save typing; correct every character.
2. Type what is printed, even if it is a spelling mistake in the original.
3. If a dot is unreadable in the scan, type your best reading and mark that
   character as ambiguous (click it with Alt).
4. `illegible`: you cannot read the whole crop with confidence. `bad_crop`: the
   crop cuts letters or contains another script. These crops are dropped.
5. Encoding variants (ه/ھ/ہ, ي/ی/ى, ڪ/ک/ك): type the canonical Sindhi form.
   Decided 2026-11-17: an engine's variant output is **not** counted in the main error metric; it is
   reported and monitored separately (`docs/decisions.md`).
6. Take a break every 45 min; record nothing from memory, always from the image.
7. **Historical print (1872–1873 items):** older type may use pre-standard letter forms
   (e.g. ڎ U+068E where modern Sindhi writes ڏ). Type the **modern standard letter the
   printed form stands for**, and add a `comment`-style note by marking the character
   ambiguous only if the dots themselves are unreadable. (Confirmed 2026-11-17, `docs/decisions.md`.)
8. Page numbers, headers, footnote markers, and crops that cut letters at the edge:
   use `bad_crop`; Latin/Devanagari text: `not_sindhi`.

## 4. Consistency (inter-pass)

One week after pass 1, `scripts/real/relabel_sample.py` picks a seeded random
10% of crops. Label them again in **blind** mode (pass 2: the labeller hides
your pass-1 text and pre-fills from a *different* engine). `scripts/real/agreement.py`
then reports:
- exact gold-text agreement and CER between passes;
- Cohen's κ of derived seq labels across all engines' candidates;
- char-level F1 of derived marks, pass 2 vs pass 1.
Disagreements are adjudicated once (pass 1 kept unless clearly wrong) and logged.

## 5. Engines (run before labelling, CPU only; no GPU hours)

Locally (Windows) Tesseract 5.5.4 comes from a conda-forge env
(`conda create -p <miniconda>/envs/tess -c conda-forge tesseract`), models in `tessdata/`:
```bash
export TESSERACT_CMD=<miniconda>/envs/tess/Library/bin/tesseract.exe TESSDATA_PREFIX=$PWD/tessdata
python -I scripts/real/run_ocr.py --engines tess_snd tess_urd tess_ara tess_fas tess_Arabic
```
Page mode: `--psm 7` (single line) with fallback 13 → 6 when empty (crops include slivers of
neighbouring lines); the mode used is stored per row.

Crop selection (`make_crops.py`, v2) finds lines with Tesseract but places word edges in white gaps of the
ink (never inside a letter); it filters by geometry (multi-line, too small, no word gaps), page rules, digits, ink density and
script (English model vs Sindhi model confidence on the crop), **never by the Sindhi engine's
own confidence**, so the set is not biased toward crops Tesseract reads well. Rejection counts:
`data/real/crop_rejections.json`. Excluded source: `ia_trumpp1872` (mostly English).

| Engine | Models | Confidence available |
|---|---|---|
| Tesseract 5 LSTM | tessdata_best `snd`, `urd`, `ara`, `fas`, `script/Arabic` (all verified to exist 2026-11-17) | word conf; per-char conf via hOCR `x_conf` (to verify on Kaggle) |
| EasyOCR | `ur`, `fa`, `ar` (no Sindhi model; unverified quality) | per-detection conf |
| Kraken | an Arabic-script model from its Zenodo repository (to pick via `kraken list`) | per-char conf |
| VLM (GPU, later) | e.g. a Qwen-VL class model, MiniCPM-OCR 1.0 | token log-probs; counts toward the 8 h baseline budget |

Engines run on the frozen crops before you label, so you never label with a
specific engine in mind.
