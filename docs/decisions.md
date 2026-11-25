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

## 2026-11-23: v2 adopted

v2 finished: session 2 resumed at 51,412 and reached 60,000 (1.43 GPU h; v2 total 9.74 GPU h).
Rule of 2026-11-22 applied to `runs/main_v2/last.pt` (step 60,000): val_font_hard AUROC **0.9997 > 0.9834** (v1)
-> **v2 is the reported model.** val_font_hard letter F1 0.989 (v1 0.884); val_font AUROC 0.9998 (v1 0.988).
v1 stays in the paper as the "5 fonts, no page realism" comparison. Ablations run on the v2 recipe
(configs/ablations_v1.json -> model_v2.json), started 2026-11-23 as parts A and B in parallel.
GPU total ~18.1 / 60 (v1 incl. debug ~8.2, v2 ~9.8, start-up overheads included approximately).

## 2026-11-23: v2 synthetic results (all sets)

`results/synth_main_v2.md`. On the untouched test fonts (Harmattan, Noto Kufi), which played no part in choosing v2:
test_font AUROC 0.9974 (v1 0.967), letter F1 0.942 (v1 0.812); test_font_hard AUROC 0.9985 (v1 0.975), letter F1 0.933
(v1 0.795). v2 beats the text-only checker on every synthetic set (all paired-bootstrap CIs above 0). Weakest groups
on unseen fonts: gaf_digraph, tah, jeem_digraph (AUROC 0.992-0.997). Synthetic sets are now near ceiling; the open
question is real scans (final evaluation).

## 2026-11-24: ablation results (synthetic)

Six runs x 16k steps on the v2 recipe (parts A and B, 7.48 + 7.11 GPU h). Table: `results/ablations.md`
(paired clustered bootstrap vs `abl_ref` on the same examples; per-run details in `results/synth_abl_*.md`).
- **Scan image is essential:** without it AUROC drops ~0.04 on every set (CIs well below 0). It still beats the
  text-only checker on real-word swaps (test_font_hard 0.958 vs 0.846), so synthetic real-word negatives still carry
  some text-only cues; to be stated in the paper.
- **Per-character head helps the sequence decision too:** removing it costs 0.03-0.05 AUROC.
- **Real-word swaps in training matter on unseen fonts:** v1 mix loses 0.021-0.030 on font sets, nothing in-font.
- **Re-render and char alignment add little on synthetic data:** small losses on unseen fonts (-0.002 to -0.004,
  CIs just below 0 on val_font_hard / test_font), none on the hard in-font sets. Kept in the model; whether they
  matter on real scans is checked in the final real evaluation (ablation checkpoints scored there, not selected).
GPU total ~32.7 / 60.

## 2026-11-24: blind relabel, word crops (pass 2)

34 word crops (`data/real/relabel_queue.txt`) relabelled blind on 2026-11-24 (local time; the export file name uses
the UTC date 2026-11-23) -> `data/real/gold_pass2.csv`; report `data/real/agreement.json`.
Pass 2 was typed with ی / ھ and RLM marks where pass 1 mostly has ي / ه (a keyboard difference; labelling rule 5),
so agreement is reported raw and **encoding-folded** (`realset.fold_encoding`: encoding groups -> canonical form,
bidi marks dropped):
| | raw | folded |
|---|---|---|
| status agreement | 1.00 | 1.00 |
| exact gold text | 0.29 | 0.65 |
| mean CER | 0.146 | 0.062 |
| error-label kappa (170 engine outputs) | 0.34 | 0.77 |
| error-label agreement | 0.82 | 0.94 |
| letter-mark F1 | 0.78 | 0.94 |
Of 12 folded disagreements, 7 are harakat or punctuation only; 5 differ in letters (baloch1961_p0077_v2w04,
bcp1873_p0012_v2w06, dharejo2002_p0045_v2w12, bcp1873_p0016_v2w04, lughat1960_p0239_v2w03).
**Adjudication (author, 2026-11-24): pass 1 kept for all.** gold.csv unchanged, real set stays frozen.
Paper reports the folded numbers as label consistency and the raw ones in the appendix.

## 2026-11-24: Jev zero-shot baseline setup (#3)

MiniCPM-OCR Jev (`openbmb/MiniCPM-OCR-Jev`, Apache-2.0) is run **as released, no fine-tuning**, through its own SDK
(transformers backend, score-only endpoint `/v1/score`). Facts from OpenBMB's docs and code (read 2026-11-24):
- Jev's render+score path renders with Chromium using CJK fonts only and no RTL handling, so we use the
  **score-only path and supply our HarfBuzz render of the candidate** (the same render our model sees).
- Default dtype is bf16, which a T4 lacks. The notebook tries float16, then float32, and keeps the first dtype whose
  scores on OpenBMB's bundled examples are within 0.02 of their reference probabilities (recorded in run_info.json).
- Output `prob_1` = P(correct); error score = 1 - prob_1. Failed items are counted and scored 0.5, never dropped.
Scope now: synthetic test_hard + test_font_hard (`kaggle/jev.ipynb`, `scripts/jev/`). Real candidates are scored only
in the final real evaluation (#5). Budget <= 2 GPU h; DEBUG run first to measure speed.

## 2026-11-25: Jev zero-shot results (synthetic)

`results/synth_jev.md`. MiniCPM-OCR Jev as released (repo commit c21a3ca, weights snapshot a0ed278), float16 on a T4
(reference examples within 0.005 of OpenBMB's values; token-for-token prompt check equal), score-only path with our
renders. 4,000/4,000 items scored, 0 failures, ~0.12 s/item.
| set | Jev | text-only | v2 | v2 - Jev [95% CI] |
|---|---|---|---|---|
| test_hard | 0.884 | 0.849 | 1.000 | +0.116 [+0.101, +0.130] |
| test_font_hard | 0.880 | 0.846 | 0.999 | +0.118 [+0.105, +0.133] |
Jev transfers partly to Sindhi (above the text-only checker, despite zh/en training) but is far below the 7M
task-specific model on dot-level real-word errors. Jev gives no per-letter output. Real-set scoring follows in #5.
Jev GPU: 0.18 (debug) + 0.12 (failed run, client path bug) + 0.39 = 0.69 h of its 2 h. GPU total ~33.4 / 60.

## 2026-11-25: blind relabel, line crops (pass 2)

8 line crops (`data/real/lines/relabel_queue.txt`) relabelled blind on 2026-11-25 -> `data/real/lines/gold_pass2.csv`;
report `data/real/lines/agreement.json`. **Protocol deviation:** done 5 days after pass 1 (2026-11-20) instead of
the planned 7 (author's choice); to be stated in the paper. Typed with the labeller's new Sindhi keyboard
(canonical-letter fix on), so raw and encoding-folded numbers barely differ.
| | raw | folded |
|---|---|---|
| status agreement | 1.00 | 1.00 |
| exact gold text | 0.25 | 0.25 |
| mean CER | 0.104 | 0.093 |
| error-label kappa (40 engine outputs) | 1.00 | 1.00 |
| letter-mark F1 | 0.91 | 0.92 |
Most differences are harakat added in pass 2 and punctuation. 3 lines differ in letters (bcp1873_p0016_v2L01 ڎ/ڏ,
mehran1959_p0123_v2L02 ب/ڀ and extra ه, mehran1959_p0134_v2L02 extra ه and ں).
**Adjudication (author, 2026-11-25): pass 1 kept for all.** Line gold unchanged, line set stays frozen. Issue #4 done.

## 2026-11-25: final real-set evaluation (run once)

Summary and tables: `docs/final_real_results.md`. One `eval_real.py --final` run per set (words, lines, and both
EasyOCR addenda), all methods at once: v2, v1, six ablations, Jev, OCR confidence, text-only, voting.
The words-addendum run crashed in data preparation before any score existed (a space with no render x-range in
EasyOCR output); fixed with `data.fill_ranges` (does not change the other three sets) and re-run once. Logged.
Results (words, 5 engines): v2 0.884 = Jev 0.885 > OCR conf 0.835 > voting 0.819 > text 0.714; dot errors only:
v2 0.935, Jev 0.879. Lines: v2 0.458 (fails), Jev 0.910. Engine selection by v2 does not beat `tess_snd`.
Decisions: v2 stays the reported model (no change after seeing real data). The tool is scoped to word crops.
Line-level training is recorded as future work, not done on this test set. Issues #5 and #3 are done.
Jev GPU 1.04 h of 2; total ~33.75 / 60.

## 2026-11-26: pip tool, calibration, batch effect, dot repair

- **Tool** (`src/sindhijev/tool.py`, `cli.py`, `pyproject.toml`; docs `docs/tool.md`): ONNX (dynamo exporter;
  the legacy exporter fixed the text length), 29.6 MB single file, parity with PyTorch < 2e-5. Scoped to word crops
  (real-set result). Weights stay local until a GitHub release is approved; PyPI name `sindhijev` was free on 2026-11-25.
- **Calibration:** temperature 1.145 fitted on synthetic val + val_hard (`results/calibration_main_v2.json`).
- **Batch effect:** BatchNorm makes padded columns non-zero, so a score depends on its batch. Effect on reported
  AUROC: <= 0.0001 on all synthetic test sets (`results/batch_effect.json`). Decision: the tool scores one item per
  forward pass; the real-set results (batch 64) are NOT re-run (it would be a second access); stated as a limitation.
- **Dot repair** (`results/dot_repair.md`): margin 0.6 chosen on val_hard; test_hard 980/1000 fixed, 1 broken;
  test_font_hard 869/1000, 2 broken. Synthetic only; not run on the real set (would need a new pre-registered run).

## 2026-11-26: published; sweep

- Hugging Face Space live (Gradio SDK; Docker is paid on this account; runs on ZeroGPU hardware with a never-called
  `@spaces.GPU` placeholder, model on CPU): https://huggingface.co/spaces/ArsalanKaleem/sindhijev. Checked via its API.
- Docs on GitHub Pages (manual workflow, custom domain of the account, HTTPS enforced):
  https://arsalankaleem.me/sindhi-JEV/ ; set as the repository website.
- Sweep fixes: README now links docs and demo and states the word-crop scope; repository description no longer
  mentions lines; `requirements.lock.txt` no longer pins `sindhinorm` to a local path (only `prepare_text.py` needs it);
  review-cost table added from stored results (no new real-set access). Issues #7 and #11 closed.
