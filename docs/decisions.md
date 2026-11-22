# Design decisions log

Decisions made by the project owner (native Sindhi speaker). Dated; later changes are appended, never edited.

## 2026-11-17: after the confusion-set review

1. **Encoding variants** (ه/ھ/ہ, ي/ی/ى, ڪ/ک/ك): **excluded from the main OCR error metric and reported
   separately.** They may be orthographic or encoding differences rather than genuine recognition errors.
   They are still **monitored** (own table: rate per engine and per variant), because consistency in Sindhi
   text matters. Implementation: `kind=encoding` groups have weight 0 (never generated as synthetic
   negatives); `align.derive_labels` tags them with their group, so evaluation can split them out.
   The reviewer notes ی and ك are Urdu/Persian/Arabic code points rarely used in Sindhi.
2. **Held-out scan fonts: Harmattan + Noto Kufi Arabic** (confirmed). They test whether the checker
   generalises to font styles it never saw in training (`configs/data_v1.json`, set `test_font`).
3. **Full-line crops: added as a separate evaluation condition**; the 1–4-word crops stay, so results can
   be compared by crop length. Lines live in `data/real/lines/` with their own crops, candidates, gold and
   freeze hashes, so the word-level set is unchanged. Reported as a separate table, never pooled.

4. **Historical letter forms** (1872–73 print, e.g. ڎ U+068E): **type the modern standard letter** the printed
   form stands for (ڎ -> ڏ). Mark a character ambiguous only if its dots are unreadable in the scan. Consequence
   to report: on historical crops, an engine that outputs the old code point counts as an error against gold.

## Confusion set v1 (same date)

Review applied by `scripts/apply_confusion_review.py`; details in `docs/confusion_set_changelog.md`.
Summary: 102/115 pairs valid. Rejected: the joined-tooth pairs across ب/ت vs ن/ڻ/ي, جه/ڄ, ف/ق and ڦ/ق,
and ك against ڪ/ک. Pair sampling weight = group weight × {H 1.0, M 0.5, L 0.2}.

## Synthetic scan check (same date)

Reviewer looked at `data/synth/test_font_neg_sheet.png` (`data/synth/scan_observations.txt`): "All examples are
readable and no dots appear unobservable." This is a visual check of one sheet, not a measurement of dot loss
under degradation; that measurement is still to do.

## Crops re-cut (same date)

Reviewer found nearly all v1 crops were bad crops (letters cut at the edges). Cause: v1 used Tesseract's
word boxes, which split Sindhi words at non-joiner gaps. v2 (`scripts/real/make_crops.py`) uses Tesseract
only to find lines; word edges are placed in white gaps measured on the ink (gap > 0.15 x line height),
and crops with page rules, digits only, or no visible word gaps are rejected. Result: 471 word crops,
80 full lines (ids carry `_v2`). v1 crops kept in `data/real_v1_crops/` (not used).
Known bias to report: crops exist only where the ink has clear word gaps; tightly set historical print
(ia_tanbih1872 13, ia_kalimat1934 18 crops) is under-represented.

## 2026-11-18: word-level real set frozen

Pass 1 labelled by the owner: 471 crops -> 337 ok, 76 illegible, 40 bad_crop, 18 not_sindhi.
- `ia_baloch1961_p0053_v2w02` was saved as ok with empty text: **dropped** (owner's decision); `derive` skips
  ok rows with empty gold and prints a warning. 336 usable crops, 1,680 labelled engine outputs.
- All pass-1 `override_json` entries (18 crops) were accidental clicks (owner did not intend overrides):
  labels derived with `--ignore-overrides`. The labeller's engine rows are now read-only unless `?override=1`.
- Latin , ; ? are mapped to Arabic ، ؛ ؟ in gold and candidates (`realset.norm`): same mark, not an OCR error.
Hashes in `data/real/FROZEN.json` (includes `candidate_labels.csv`). No further access before final evaluation
except the blind 10% relabel (pass 2, due on or after 2026-11-24).

## 2026-11-20: Phase 3 plan approved

Owner approved `docs/phase3_plan_and_research.md` in full:
1. **Decision gate** on `val_hard`: after ~3 GPU h of main training, model AUROC must beat the text-only checker
   (0.849) by >= +0.03 with the paired-bootstrap 95% CI lower bound > 0; else stop and diagnose.
2. **All experiments**: real-word training mode and hard sets; engine selection; dot repair by search; review-cost
   curve; calibration; MiniCPM-OCR Jev zero-shot (<= 2 GPU h, baseline budget); extra OCR engines (EasyOCR on
   CPU, later a VLM OCR) as a **frozen addendum** `data/real/addendum_v1/` (union of the frozen candidates and the
   new engines, labels derived from the existing frozen gold; the original frozen files are not modified);
   ablations (no scan / no render / no char-alignment / no char head / v1 mix); pip tool after final evaluation.

## 2026-11-20: engine addendum v1 frozen

`data/real/addendum_v1/` = frozen real set + EasyOCR 1.7.2 (CPU) `easyocr_ur`, `easyocr_fa` on all 471 crops.
Labels derived from the same frozen gold (no new labelling): 336 usable crops x 7 engines = 2,352 labelled
outputs. EasyOCR is weak here (fully correct: ur 21/336, fa 25/336; 198 of 942 outputs empty: no text detected),
so it mainly adds error diversity. Original `data/real/FROZEN.json` verified unchanged before and after.

## 2026-11-20: full-line set frozen

Owner labelled all 80 lines (pass 1): 74 ok, 4 illegible, 2 bad_crop; no overrides, no empty text; median 62 s/line.
74 x 5 Tesseract engines = 370 labelled outputs, frozen in `data/real/lines/FROZEN.json`. Whole lines are rarely
fully correct (tess_snd 12/74, all engines 15/370), so for lines the char-level metrics (F1, top-1 localisation)
are the main numbers; line-level AUROC will have wide CIs. Blind relabel queue: 8 lines (pass 2, on/after 2026-11-27).
EasyOCR addendum for lines: `data/real/lines/addendum_v1/` (+160 outputs), frozen separately.

## 2026-11-21: main run session 1, gate PASSED

Session 1 (Kaggle T4, 3.20 GPU h logged + ~0.25 h start-up): steps 0-26,001 of 60,000, 0.44 s/step.
val_hard AUROC 0.9918 at step 4k, 0.9999 at step 24k; text-only checker 0.849 -> gate (>= 0.879) passed by
~+0.15 (formal paired-bootstrap CI to be computed locally from the checkpoint).
Caveat recorded: the model saturates the synthetic sets, so synthetic validation can no longer rank checkpoints.
**Rule fixed now, before any real-set result: the reported model is `last.pt` at the end of the schedule (step
60,000), not best.pt.** Real-set performance is the open question; it is measured once, at final evaluation.
GPU so far: ~3.8 / 60 (debug ~0.3, main ~3.45).

## 2026-11-22: main run finished; synthetic results

Session 2 resumed at step 26,001 and ended at 60,000 (4.15 GPU h). Main block used ~7.9 h (logged 7.35 + start-up);
project total ~8.2 / 60. Reported model = `runs/main/last.pt` (rule of 2026-11-21).
Synthetic results (`results/synth_main.md`): gate formally passed on val_hard, model 1.000 vs text 0.849,
delta +0.151 [95% CI +0.133, +0.167]. Training fonts: ~1.000 everywhere. **Held-out fonts are the weakness:**
test_font AUROC 0.967 (text 0.952, delta +0.015 [+0.006, +0.023]), char F1 0.81; test_font_hard 0.975, char F1 0.79.
Open question for the owner: train a v2 with more font diversity before the ablations (see chat 2026-11-22).
Honesty note: this gap was seen on a synthetic TEST set; any v2 must be chosen on a separate held-out-font
VALIDATION set (a third font family), and both v1 and v2 numbers get reported.

## 2026-11-22: v2 prepared (data_v3 / model_v2)

Data v3: 9 training fonts (+ IBM Plex Sans Arabic, Kufam, Jomhuria, Katibeh; all pass check_fonts) and page realism
(neighbour-line slivers, bleed-through, stretch, uneven lighting, binarisation). v1/v2 data reproduce exactly (checked).
Unseen-font validation font: Mirza (Naskh); sets val_font / val_font_hard (2,000 each).
**v1 on val_font_hard: AUROC 0.9834** (char F1 0.884). Rule fixed before v2 training: v2 becomes the reported
model only if its val_font_hard AUROC (last.pt) is higher than 0.9834. Test sets and the real set are not used.

## 2026-11-23: v2 session 1

v2 (data_v3, model_v2) session 1: 8.31 GPU h, stopped at step 51,412 of 60,000 (data generation slower: ~0.56 s/step).
val_font_hard AUROC at step 51,412: **0.9997** (letter F1 0.989) vs v1 final 0.9834 (letter F1 0.884); val_font 0.9998
vs 0.988. Per the 2026-11-22 rule the adoption check uses last.pt at step 60,000 -> one resume session (~1.4 h).
Ablations shortened to 16k steps each (6 runs ~15 GPU h at v2 speed), same budget for every variant.
GPU total ~16.6 / 60.
