# Phase 3: what was built, what I decided, and what needs your OK (2026-11-18)

GPU hours used: **0 / 60** (all of tonight's work ran on CPU).

## 1. One important finding first

The text-only baseline (character 5-gram LM in both reading directions + a 49k-word dictionary, trained on the
train text only) already reaches **AUROC 0.95–0.97** on the v1 synthetic sets (val / test_iid / test_font).
Reason: only **12%** of the synthetic dot swaps produce a real word; the rest produce non-words that a spell checker
catches without looking at the image. So:

- the planned decision gate ("clearly above the text-only baseline on hard negatives") was nearly unpassable;
- worse, a model trained on v1 data could learn spelling instead of looking at the dots.

**Decision (done):** a new corruption type, **real-word dot swaps**: the swapped word is itself in the dictionary
(e.g. ٻار → بار). New fixed sets `val_hard`, `test_hard`, `test_font_hard` (2,000 each: half correct, half
real-word swaps). The text-only checker drops to **AUROC 0.85** there. The training mix (configs/data_v2.json) now
has 15% real-word swaps. The v1 sets stay unchanged (reproducible, 30/30 checked).

**New decision gate:** after ~3 GPU h of main training, `val_hard` AUROC must beat the text-only checker
(0.849) by **≥ +0.03, with the lower end of the paired bootstrap 95% CI above 0**. Otherwise stop and diagnose.

## 2. What was built tonight (all tested on CPU; 35 tests pass)

| Piece | File | Notes |
|---|---|---|
| Model | `src/sindhijev/model.py` | 7.1M params. CNN scan encoder + CNN render encoder → tokens; one query per candidate code point, seeded with the render features at that letter's x-position; 4-layer transformer decoder; sequence head (CLS) + per-character head. Ablation switches: `use_scan`, `use_render`, `char_align`, `char_head`. |
| Data | `src/sindhijev/data.py` | vocabulary, height-48 preprocessing, batching with masks; on-the-fly generation, fixed sets, shards, real set. |
| Training | `scripts/train.py` | fp16, checkpoint every epoch (`last.pt`, `best.pt` by `val_hard` AUROC), automatic resume, per-session time limit, GPU-hours log (`gpu_time.json`), refuses to run past the budget. Smoke test: loss falls, resume continues at the right step, the model overfits 64 examples to AUROC 1.0. |
| Shards | `scripts/make_shards.py` | pre-generate pairs on CPU if the GPU waits for data (~9 KB/pair, 125 pairs/s/core). |
| Metrics | `src/sindhijev/metrics.py` | AUROC/AUPRC (match scikit-learn exactly), routing at 10/20/30%, char P/R/F1, top-1 localisation, crop-clustered bootstrap CIs, paired bootstrap, ECE, engine selection. |
| Baselines | `src/sindhijev/baselines.py`, `scripts/baselines.py` | (a) OCR confidence, (b) text-only checker, (c) multi-engine voting; non-word vs real-word split. |
| Synthetic eval | `scripts/evaluate.py` | model vs text checker on all 6 synthetic sets, paired CIs, per group / font / error type. |
| Final real eval | `scripts/real/eval_real.py --final` | refuses without `--final` or if the frozen hashes changed; logs each run in `docs/real_set_access_log.md`; all 4 methods, per engine / stratum, non-word vs real-word, dot errors only, encoding variants separate, engine selection, calibration. Tested on a scratch copy of the demo set only. |
| Kaggle | `kaggle/train.ipynb`, `kaggle/make_bundle.py` | bundle = 75 MB (`dist/sindhijev_kaggle.zip`), no real data inside. |

**Decision (done): GPU is used only for training.** All evaluation (synthetic and real) runs on your laptop CPU
from the downloaded checkpoint: 0 GPU hours.

### GPU plan (60 h)

| Block | Hours | What |
|---|---|---|
| Debugging | 5 | session 1: 300-step speed test (~0.3 h); fix data bottleneck if any (shards) |
| Main | 25 | 3 sessions × ~8.3 h; gate check after ~3 h |
| Ablations | 15 | 5 runs × 3 h: no scan, no render, no char-alignment, no char head, v1 mix (no real-word swaps) |
| Baselines | 8 | MiniCPM-OCR Jev zero-shot (~2 h); optional VLM OCR as a 6th engine (~2 h) |
| Reserve | 7 | – |

## 3. Research: what is new since the Phase 0 audit (searched 2026-11-18)

- **Render-and-compare verification is now common in big systems**: OCR-EDR (arXiv 2609.03445), Xiaomi-OCR-0
  (arXiv 2609.36136: renders a candidate back and compares with the region, up to 3 verification rounds, then
  routes to humans), Multimodal OCR (arXiv 2603.13032: render-based checks in its data pipeline), MiniCPM-OCR Jev
  (2B verifier, AUROC 0.936 on JevOCRBench per secondary sources, Apache-2.0). None targets Sindhi or dots,
  and all are 2B+ parameter models (except as data pipelines). Our angle stays: **small (7M, CPU), dot-level,
  script-specific, real Sindhi benchmark, real-word error analysis.** Add Xiaomi-OCR-0 and Multimodal OCR to the
  related-work table.
- **Sindhi data**: a 2025 ESWA paper ("End-to-end contextual-aware deep learning pipeline for Sindhi text
  recognition") reports a real printed set, RW-Sindhi (20k word / 17k line images); **availability unverified**.
  Proxima AI published a synthetic Sindhi OCR set (278k pairs, CC BY-NC 4.0) on Mozilla Data Collective. Several
  small Hugging Face sets exist (Sanapalijo/Sindhi_OCR, DanishMahdi/snd-headline-ocr, gated). I found no public
  Sindhi OCR **error-detection** dataset → our benchmark is still the first I know of (search not exhaustive).
- **GlotOCR Bench (ICML 2026)**: Arabic-script errors are dominated by visually similar characters and
  diacritics (supports our motivation; whether it scores Sindhi separately is unverified).
- **PyPI**: no Sindhi OCR or OCR-checking package found (Sindhi NLP packages exist: sindhinltk,
  aadil-nazar-sindhi-nlp). A pip tool would be the first (search not exhaustive).

## 4. Improvements: decisions

Ranked by reward / effort. **Done** = built tonight. **Decided** = I will do it. **Needs OK** = yours to approve.

| # | Idea | Effort | Reward | Status |
|---|---|---|---|---|
| 1 | Real-word dot errors: training mode + hard sets + non-word/real-word split of all results | done | high: proves the image matters; honest gate | **Done** |
| 2 | **Engine selection**: per crop keep the engine output the checker trusts most → report CER and exact match vs each single engine and the oracle | done (code) | high: turns "detector" into "better OCR text" with no extra labels | **Done** (runs in final eval) |
| 3 | **Dot repair by search**: for the most suspicious letter, try the other letters of its confusion group, re-score each with the checker, keep a fix only if it clearly wins | ~0.5 day, CPU | high: a corrector for free; strong demo | **Decided**, after training |
| 4 | Review-cost curve: % of errors caught vs % of crops reviewed, in minutes (your median labelling time is 14 s/crop) | ~1 h | medium: practical framing for the paper | **Decided** |
| 5 | Calibration: report ECE; temperature-scale on synthetic val for the tool | done / 1 h | medium | **Done** (ECE) |
| 6 | MiniCPM-OCR Jev zero-shot on the real set | ~2 GPU h | high for reviewers ("did you compare with Jev?") | **Decided** (baseline budget) |
| 7 | More OCR engines on the real crops (EasyOCR ur/fa on CPU; a VLM OCR on GPU). Labels come from your gold text automatically, so **no new labelling** | CPU / ~2 GPU h | high: shows the checker is not tuned to Tesseract | **Needs OK**: adds a frozen *addendum* (new files, existing frozen files untouched) |
| 8 | Full-line results (80 lines, already cut) | your labelling | medium: crop-length comparison you asked for | waiting for your labels |
| 9 | **pip tool** `sindhijev` (name availability unverified): `sindhijev check crop.png "text"` → score + suspicious letters; `sindhijev pick crop.png` → runs installed Tesseract models, returns the best output with flagged letters. ONNX model (~15–30 MB) downloaded on first use from Hugging Face; CPU only; OFL font bundled. Plus a free Hugging Face Space demo. | 1–2 days | high: usable artefact, demo-track paper, citations | **Decided**, after the final evaluation (weights must be final) |
| 10 | Ask the RW-Sindhi authors for their real printed set as a second real test set | an email | medium–high (bigger real eval) | **Needs OK**: you send the email |
| – | Not doing: training our own OCR, handwriting model, VLM fine-tuning, Urdu/Saraiki extension (future work) | high | low for this paper | **Rejected** |

## 5. Needs your answer tomorrow

1. ~~Gate change~~ approved 2026-11-20.
2. ~~Item 7~~ approved 2026-11-20.
3. **Item 10**: will you email the RW-Sindhi authors? (I can draft it.)
4. **Kaggle**: upload `dist/sindhijev_kaggle.zip` as a private dataset `sindhijev-code`, open `kaggle/train.ipynb`
   with GPU on, run with `DEBUG = True` (~20 min GPU) and send me the printed speed line and `nvidia-smi` output.
5. Still pending from before: label the 80 lines; the blind 10% relabel on/after 24 Nov.

## Sources

- [OCR-EDR (arXiv 2609.03445)](https://arxiv.org/abs/2609.03445)
- [Xiaomi-OCR-0 Technical Report (arXiv 2609.36136)](https://arxiv.org/html/2609.36136v1)
- [Multimodal OCR: Parse Anything from Documents (arXiv 2603.13032)](https://arxiv.org/html/2603.13032v1)
- [MiniCPM-OCR README (GitHub)](https://raw.githubusercontent.com/OpenBMB/MiniCPM-OCR/main/README.md); [Jev summary (DataLearner)](https://www.datalearner.com/ai-models/pretrained-models/minicpm-ocr-jev)
- [End-to-end contextual-aware deep learning pipeline for Sindhi text recognition (ESWA)](https://www.sciencedirect.com/science/article/abs/pii/S0957417425045750)
- [Sindhi OCR by Proxima AI (Mozilla Data Collective)](https://mozilladatacollective.com/datasets/cmrywapqg015ho907f9q9hvka)
- [Sanapalijo/Sindhi_OCR](https://huggingface.co/datasets/Sanapalijo/Sindhi_OCR); [DanishMahdi/snd-headline-ocr](https://huggingface.co/datasets/DanishMahdi/snd-headline-ocr)
- [GlotOCR Bench (arXiv 2604.12978)](https://arxiv.org/html/2604.12978v1)
- [aadil-nazar-sindhi-nlp (PyPI)](https://pypi.org/project/aadil-nazar-sindhi-nlp/)
