# Sindhi-Jev

**Check Sindhi OCR text against the scan, and find the letters whose dots are wrong.**

[![PyPI](https://img.shields.io/pypi/v/sindhijev?color=1a5fd0)](https://pypi.org/project/sindhijev/)
[![tests](https://github.com/ArsalanKaleem/sindhi-JEV/actions/workflows/tests.yml/badge.svg)](https://github.com/ArsalanKaleem/sindhi-JEV/actions/workflows/tests.yml)
[![docs](https://img.shields.io/badge/docs-arsalankaleem.me%2Fsindhi--JEV-1a5fd0)](https://arsalankaleem.me/sindhi-JEV/)
[![live demo](https://img.shields.io/badge/%F0%9F%A4%97%20live%20demo-Hugging%20Face-ffcc4d)](https://huggingface.co/spaces/ArsalanKaleem/sindhijev)
[![licence](https://img.shields.io/badge/licence-Apache--2.0-555)](LICENSE)

**[Documentation](https://arsalankaleem.me/sindhi-JEV/)** ·
**[Live demo](https://huggingface.co/spaces/ArsalanKaleem/sindhijev)** ·
**[PyPI](https://pypi.org/project/sindhijev/)** ·
**[Results](docs/final_real_results.md)** ·
**[Release v0.1.0](https://github.com/ArsalanKaleem/sindhi-JEV/releases/tag/v0.1.0)** ·
**[Roadmap](https://github.com/ArsalanKaleem/sindhi-JEV/issues/14)**

Sindhi letters are often told apart only by their dots: ب ٻ ڀ پ, ن ڻ, ج ڄ, د ڏ ڌ. OCR engines frequently get
the letter shape right and the dots wrong, and many of those mistakes are still real words, so a spell checker cannot
see them. Sindhi-Jev is a 7M-parameter model that looks at the **scanned word crop and the OCR text together**,
estimates how likely the text is wrong, and points to the letters whose dots look wrong. It runs on an ordinary CPU in
about 20 ms per crop.

To our knowledge (a non-exhaustive literature search, [docs/phase0_audit.md](docs/phase0_audit.md)), this is the
first OCR error-detection work for Sindhi, with the first real Sindhi OCR test set labelled letter by letter.

## Install and use

```bash
pip install sindhijev
```

```bash
sindhijev check word.png "ڪلوواٽ اهي"
# p_wrong=1.000  ڪلوواٽ اهي
# suspicious letters (positions [7]): ا

sindhijev repair word.png "ڪلوواٽ اهي"
# ڪلوواٽ آهي
#   ڪلوواٽ اهي -> ڪلوواٽ آهي  p_wrong 1.000 -> 0.000

sindhijev pick word.png --cand "ڪلوواٽ اهي" --cand "ڪلوواٽ آهي"   # add --tesseract to run installed Tesseract models
```

```python
from sindhijev.tool import Checker

ck = Checker()
r = ck.check("word.png", "ڪلوواٽ اهي")
print(r["p_wrong"], [r["text"][i] for i in r["suspicious"]])
```

On first use the model (`sindhijev_v2.onnx`, ~30 MB) is downloaded once from the
[GitHub release](https://github.com/ArsalanKaleem/sindhi-JEV/releases/tag/v0.1.0) and checked by SHA-256. No GPU and
no PyTorch are needed. Full reference: [command line](https://arsalankaleem.me/sindhi-JEV/cli.html),
[Python API](https://arsalankaleem.me/sindhi-JEV/python-api.html). No install at all:
[live demo](https://huggingface.co/spaces/ArsalanKaleem/sindhijev).

## Results

Measured **once** on a frozen, hand-labelled set of real scans: 336 word crops from 14 printed Sindhi sources, each
read by 5 Tesseract models (1,680 OCR outputs). AUROC = how well a method ranks wrong OCR outputs above correct ones
(1.0 perfect, 0.5 chance). Brackets: paired, crop-clustered bootstrap 95% CI for v2 minus the method.

| Method | All errors | v2 − method | Dot errors only |
|---|---|---|---|
| **Sindhi-Jev v2** (7M, this project) | **0.884** | | **0.935** |
| MiniCPM-OCR Jev (2B, zero-shot) | 0.885 | −0.001 [−0.038, +0.037] | 0.879 |
| OCR engine confidence | 0.835 | +0.049 [+0.014, +0.083] | 0.817 |
| Agreement between engines | 0.819 | +0.065 [+0.030, +0.100] | 0.813 |
| Text-only checker (char 5-gram LM + dictionary) | 0.714 | +0.170 [+0.130, +0.212] | 0.705 |

- **Matches a 2B general verifier** overall, and is the best method measured on **dot errors**, the task it was built
  for. Adding 2 EasyOCR engines (2,352 outputs) gives the same picture: v2 0.896, Jev 0.898; dot errors 0.940 vs 0.883.
- **Every part matters on real scans** (ablations, paired vs v2): no scan image −0.137, no per-letter head −0.118,
  no letter alignment −0.088, no rendered candidate −0.062, no real-word swaps in training −0.061.
- **Limits, reported as measured:** on full **lines** it fails (0.458; Jev 0.910); choosing the output it trusts most
  does not beat always using Tesseract `snd` (CER 0.215 vs 0.181); letter flags are a hint (top flag right 84% of the
  time on wrong outputs, letter F1 0.47); `repair` is measured on synthetic data only (fixes 980/1,000 real-word dot
  swaps in training fonts, 869/1,000 in unseen fonts).
- **Other limits:** the dot-confusion set and the real labels were reviewed by one native speaker (the author);
  training text is Sindhi Wikipedia only; printed Naskh-style type only (handwriting and Nastaliq untested).

Everything else: [docs/final_real_results.md](docs/final_real_results.md) (real set),
[results/](results/) (all result files), [docs/decisions.md](docs/decisions.md) (every decision, with dates).

## How it works

The OCR text is rendered with HarfBuzz in Noto Naskh Arabic. Two small CNNs turn the scan and the render into token
sequences (one token per 4 px of width); a 4-layer transformer decoder reads them with one query per letter, each
initialised from the render tokens under that letter, plus one query for the whole text. Outputs: P(text wrong) and
P(letter wrong) for every letter.

Training data is fully synthetic, from Sindhi Wikipedia: true text drawn in 9 open-licence fonts and degraded like a
scan; candidates are correct or corrupted with dot swaps from a native-reviewed confusion set, substitutions,
insertions and deletions, including **real-word swaps** (one dictionary word turned into another), which force the
model to use the image. Fonts are split three ways (train / model selection / test only). The reported model was
trained for 60,000 steps on one Kaggle T4 (9.7 GPU hours). Details:
[How it works](https://arsalankaleem.me/sindhi-JEV/how-it-works.html).

## The real test set

- 471 word crops and 80 line crops cut at ink gaps from 14 printed sources (born-digital, printed, historical); 336
  words and 74 lines usable after labelling. Protocol: [docs/labeling_protocol.md](docs/labeling_protocol.md).
- Labels are derived per code point from hand-typed gold text; encoding variants (ه/ھ/ہ, ي/ی/ى, ڪ/ک/ك) are reported
  separately, not counted as errors.
- **Frozen** with SHA-256 hashes (`data/real/FROZEN.json`); `scripts/real/eval_real.py` refuses to run without
  `--final` and on any changed file. Every access is logged: [docs/real_set_access_log.md](docs/real_set_access_log.md).
- Blind relabel of 10% (author, after 7 days for words, 5 for lines): error-label kappa 0.77 (words) and 1.00 (lines),
  letter-mark F1 0.94 and 0.92, net of keyboard code-point differences.
- Real scan images are not in this repository; each source's licence status is in `data/real/sources.csv`.

## Reproduce

```bash
py -3.12 -m venv .venv                       # Linux: python3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.lock.txt
.venv/Scripts/python -m pytest tests
```

| Step | Command |
|---|---|
| Text spans from a Sindhi Wikipedia dump (uses [`sindhinorm`](https://pypi.org/project/sindhinorm/) 0.1.0) | `python scripts/prepare_text.py --corpus sd_wiki.txt` |
| Train-split lexicon for real-word swaps | `python scripts/build_lexicon.py` |
| Fixed synthetic validation and test sets | `python scripts/build_pairs.py --config configs/data_v3.json --sets val val_hard val_font_hard test_iid test_font test_hard test_font_hard` |
| Train (Kaggle: `kaggle/train.ipynb` + `python kaggle/make_bundle.py`) | `python scripts/train.py --config configs/model_v2.json --out runs/main_v2` |
| Synthetic evaluation, ablations, text-only baseline | `scripts/evaluate.py`, `scripts/ablation_table.py`, `scripts/baselines.py text` |
| MiniCPM-OCR Jev baseline (Kaggle T4) | `kaggle/jev.ipynb`, `scripts/jev/` |
| Final real-set evaluation (run once) | `python scripts/real/eval_real.py --final ...` |
| Calibration and ONNX export | `python scripts/calibrate.py`, `python scripts/export_onnx.py` |
| Docs site / Hugging Face Space | `python site_src/build.py`, `python space/make_space.py` ([docs/publishing.md](docs/publishing.md)) |

Compute used: **~33.6 of a 60 GPU-hour budget** (T4): v1 ~8.2, v2 9.7, six ablations 14.6, Jev baseline 1.0.
Rendering uses HarfBuzz and FreeType directly (no libraqm), so output is identical on Windows and Linux. On Windows,
set `PYTHONUTF8=1` to print Sindhi in the console.

<details>
<summary>Data details: splits and label rules</summary>

- **Splits by sentence:** each normalised sentence is hashed (salt `sindhijev-v1`) into train 90% / val 5% / test 5%;
  all spans of a sentence share its split; duplicate sentences are dropped.
- **Splits by font:** 9 fonts for training, Mirza for model selection, Harmattan and Noto Kufi Arabic only for the
  final synthetic test. Candidates are always rendered in Noto Naskh Arabic, as a deployed checker would be.
- **Labels** are per code point of the candidate. Swap or substitution: the new unit's code points are 1. Insertion:
  the inserted code point is 1. Deletion: the code point right after the gap is 1 (before it, at the end). A corrupted
  candidate that renders pixel-identical to the truth in the scan font is relabelled correct.

</details>

## Repository layout

| Path | Contents |
|---|---|
| `src/sindhijev/` | package: alphabet, renderer, data, model, metrics, baselines, real-set guard, CPU tool and CLI |
| `scripts/` | data building, training, evaluation, calibration, export; `scripts/real/` real-set tools; `scripts/jev/` Jev baseline |
| `configs/` | data and model configs (`data_v3.json`, `model_v2.json` are the reported ones) |
| `data/` | confusion set, synthetic set manifests, real-set labels (no real images) |
| `results/` | every reported number, as JSON and Markdown |
| `docs/` | novelty audit, decisions, labelling protocol, final results, tool and publishing notes |
| `site_src/`, `space/`, `tools/` | docs site, Hugging Face Space, local demo server and labeller |
| `kaggle/` | notebooks and bundle builders for GPU runs |

## Status

Done: novelty audit, synthetic data, real benchmark (frozen), models v1 and v2, ablations, MiniCPM-OCR Jev baseline,
blind relabel, final real evaluation, calibration, review-cost analysis, pip package, live demo and docs.
Open ([roadmap](https://github.com/ArsalanKaleem/sindhi-JEV/issues/14)): paper draft, Zenodo DOI, licence check before
releasing any real images, dot repair on real scans (needs a new pre-registered run), optional handwriting test.

## Citation and licence

A paper is in preparation; until then please cite this repository (Arsalan Kaleem, *Sindhi-Jev*, 2026,
https://github.com/ArsalanKaleem/sindhi-JEV).

- **Code and model:** [Apache License 2.0](LICENSE).
- **Fonts:** SIL Open Font License; licence texts next to each font in `fonts/`.
- **Data:** [docs/DATA_LICENSES.md](docs/DATA_LICENSES.md). Sindhi Wikipedia text is CC BY-SA.

Sindhi-Jev is an independent project. It reuses the render-and-compare idea of OpenBMB's MiniCPM-OCR Jev and is not
affiliated with OpenBMB.
