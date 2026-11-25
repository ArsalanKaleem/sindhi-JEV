# Final real-set evaluation (run once, 2026-11-25)

Reported model: **v2** (`runs/main_v2/last.pt`). Same run, same candidates: v1, the six ablations, MiniCPM-OCR Jev
(zero-shot, as released) and the three baselines. Thresholds from synthetic validation only; nothing tuned on real data.
Full tables: `results/real_main_v2_{words,words_addendum,lines,lines_addendum}.md`. Access log:
`docs/real_set_access_log.md` (one crash before any score was computed, re-run once; logged).

Main labels exclude encoding-variant differences (ه/ھ/ہ, ي/ی/ى, ڪ/ک/ك; decision 2026-11-17). AUROC = how well a method
ranks wrong OCR outputs above correct ones. CIs: crop-clustered bootstrap (2000); "v2 − X" is paired on the same items.

## Words (336 crops)

| method | words, 5 Tesseract (1,680) | v2 − X [95% CI] | words + EasyOCR, 7 engines (2,352) | v2 − X [95% CI] |
|---|---|---|---|---|
| **v2 (7M, ours)** | **0.884** | | **0.896** | |
| Jev (2B, zero-shot) | 0.885 | −0.001 [−0.038, +0.037] | 0.898 | −0.003 [−0.036, +0.033] |
| OCR confidence | 0.835 | +0.049 [+0.014, +0.083] | 0.821 | +0.075 [+0.040, +0.108] |
| voting (other engines) | 0.819 | +0.065 [+0.030, +0.100] | 0.795 | +0.101 [+0.067, +0.134] |
| text-only (char LM + dictionary) | 0.714 | +0.170 [+0.130, +0.212] | 0.699 | +0.196 [+0.158, +0.236] |
| v1 (5 fonts, no page realism) | 0.860 | +0.024 [+0.002, +0.048] | 0.867 | +0.028 [+0.008, +0.051] |

**Dot errors only** (correct outputs vs outputs whose error is a reviewed dot confusion; 358 wrong on 5 engines):
v2 **0.935** [0.908, 0.957], Jev 0.879 [0.842, 0.911], OCR confidence 0.817, voting 0.813, text-only 0.705
(7 engines: v2 0.940, Jev 0.883). CIs here are per method (not paired).
**Real-word errors** (239): v2 0.842, Jev 0.848, OCR confidence 0.769, voting 0.752, text-only 0.453.

### Ablations on real words (v2 recipe, 16k steps each; paired vs v2)

| removed | AUROC (5 engines) | v2 − ablation [95% CI] |
|---|---|---|
| nothing (`abl_ref`, shorter training) | 0.866 | +0.018 [−0.002, +0.039] |
| scan image | 0.747 | +0.137 [+0.097, +0.174] |
| per-letter head | 0.766 | +0.118 [+0.078, +0.157] |
| char-aligned render pooling | 0.796 | +0.088 [+0.054, +0.125] |
| re-rendered candidate | 0.822 | +0.062 [+0.036, +0.090] |
| real-word swaps in training | 0.823 | +0.061 [+0.034, +0.087] |

On synthetic data the render and the alignment looked nearly free; on real scans every component matters.

### Letters, selection, calibration (words, 5 engines)
- Top-1 localisation (most suspicious letter is a wrong one): v2 0.838; on dot errors 0.953.
- Letter F1 (synthetic-val threshold): v2 0.467, voting 0.658. On dot errors: v2 0.589, voting 0.746.
- Engine selection (keep the output v2 trusts most per crop): CER 0.215, exact 0.423. **Worse than always using
  `tess_snd`** (CER 0.181, exact 0.443); oracle CER 0.131. Jev selection: CER 0.234.
- Calibration: ECE 0.118 (7 engines: 0.108).

## Full lines (74 crops): negative result
v2 AUROC 0.458 [0.289, 0.616] on 370 outputs (7 engines: 0.489), i.e. chance; Jev 0.910; OCR confidence 0.781.
Only 17 of 370 line outputs are fully correct, so the estimate is noisy, but v2 is clearly below Jev (paired
−0.451 [−0.625, −0.291]) and below v1 and `abl_ref`. The model was trained on 1–4-word crops (max width 1024 px at
height 48); full lines are out of its training distribution. The tool is therefore documented for word-level crops.

## What this means
1. On word crops, the 7M model matches a 2B general verifier and beats every classical baseline; on dot errors, the
   task it was built for, it is the best method measured.
2. It fails on full lines; Jev does not. Line-level training is the clearest next step.
3. Its per-crop scores do not yet pick better text than the best single engine; flagging (where to look) works
   better than selection (what to keep).
