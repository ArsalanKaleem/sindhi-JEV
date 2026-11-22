# Sindhi-Jev — Phase 0: novelty and fact audit (2026-11-17)

Status: **awaiting your approval (GATE 0).** GPU hours used: **0 / 60**.

Legend: **verified** = primary source read in this audit (README/model card/paper page).
**partial** = abstract or search snippet only. **unverified** = not opened; do not cite yet.

## 1. Does OpenBMB "Jev" exist? Yes (verified)

Sources read: github.com/OpenBMB/MiniCPM-OCR (README, `docs/minicpm_ocr_jev_en.md`,
`docs/finetune_jev_en.md`), huggingface.co/openbmb/MiniCPM-OCR-Jev-gguf.

| Item | Fact |
|---|---|
| Released | 2026-10-27, together with MiniCPM-OCR 1.0 |
| What | "OCR quality assessment model": scores an existing OCR candidate 0–1; does not recognise text |
| Inputs | region crop + candidate text + category (text/formula/table); candidate is rendered internally, or caller supplies the render |
| Output | **one scalar per candidate. No character-level localisation** (finetune doc says so explicitly) |
| Size / base | 2B, from Qwen3.5-2B; lm_head replaced by a binary classification head |
| Languages | Chinese and English only. Arabic/Urdu/Persian/Sindhi: not mentioned |
| Training negatives | real wrong OCR outputs + rule-based perturbations re-rendered; doc warns synthetic-only "fails on real OCR errors" |
| Benchmark | JevOCRBench (HF `openbmb/JevOCRBench`), AUROC×100 = 93.6 (Qwen3.5-397B: 93.8) |
| License | Apache-2.0 (code + weights) |
| Paper | none found (no arXiv/tech report linked). Expansion of "Jev": not stated |

## 2. Closest prior work

| Work | Status | Render candidate? | Char-level localisation? | Scripts | Small? | Relevance |
|---|---|---|---|---|---|---|
| **OCR-EDR / DocEDR** (Zhao et al., Tencent, arXiv 2609.03445, Sep 2026) | verified (abstract+HTML, partly) | **Yes** (image + text + render; ablates I+p, I+R, I+p+R) | **Yes** (char-level boundaries, ±3 tolerance) + error type + repair | Chinese/English shown; Arabic-script not stated | No (Qwen3.5-9B, SFT+GRPO) | **Closest. Already does render-aware verification + localisation.** No OCR-confidence, text-only-LM or voting baselines reported |
| **MiniCPM-OCR Jev** (OpenBMB, Oct 2026) | verified | Yes | No | zh/en | 2B | Sequence-level render-and-verify; usable as baseline (e) |
| **US 5,889,897** (Medina, filed 1997, expired) | verified | Yes (regenerate bitmap from font, compare to scan) | No | — | — | The core idea is ~30 years old; we cannot claim "render-and-verify" as new |
| ConfBERT, "Confidence-Aware Document OCR Error Detection" (Hemmer et al., arXiv 2409.04117) | partial (abstract) | No (text + OCR confidence) | not stated in abstract | Latin | small | Baseline family (a)+(b) |
| Arabic vs Latin HTR gap analysis (arXiv 2606.18884) | partial (snippets) | — | — | Arabic | — | Reports dot-based vs shape-based confusion groups; ~29–30% of Arabic substitutions between similar chars. Good motivation citation **after** reading |
| GlotOCR Bench (arXiv 2604.12978) | partial | — | — | many | — | Arabic-script OCR lags due to similar characters |
| From Press to Pixels, Urdu newspaper benchmark (LREC 2026, arXiv 2505.13943) | partial | — | error analysis only | Urdu | — | Real Urdu newspaper set; diacritic/substitution errors as main failures |
| Arabic OCR post-correction, confusion matrix + shallow LM (Kissos & Dershowitz) | partial | No | word-level | Arabic | small | Text-side correction baseline |
| RoundTripOCR, Sanskrit post-OCR sets | partial | No | — | Devanagari | — | Synthetic post-OCR data recipe; no Sindhi |
| Sindhi OCR (seg., char recognition; USindh/QUEST journals) | partial | — | — | Sindhi | — | No Sindhi OCR-verification or post-correction dataset found |

Not found (in ~8 searches, so not exhaustive): any OCR verifier or post-corrector targeting
Arabic-script dots (i'jam) explicitly; any Sindhi OCR error-detection dataset.

## 3. What is and isn't new

**Not new — do not claim:**
- Render-the-candidate-and-compare (1997 patent; Jev; OCR-EDR).
- Sequence-level OCR correctness scoring (Jev, OCR-EDR, ConfBERT).
- Character-level error localisation from image + text + render (OCR-EDR, 9B, zh/en).

**Plausibly new (if results hold):**
1. Dot-aware verification for a Perso-Arabic script, with a **native-reviewed dot-confusion set**
   as the negative generator and as the reporting axis.
2. A **small** (≤60M) verifier vs 2B/9B ones — "does a tiny, script-specific verifier match a
   general 2B verifier on dot errors?"
3. First **real Sindhi OCR-verification test set** with per-character error labels (300–500 crops).
4. Head-to-head against **cheap baselines the big papers skip**: OCR confidence, char n-gram LM +
   dictionary, multi-engine voting. Key sub-question: dot swaps often yield *valid words*
   (real-word errors), which text-only checkers cannot see — split all results by
   non-word vs real-word error.
5. Zero-shot transfer test of Jev (zh/en-trained) to Sindhi: a cheap, informative baseline.

## 4. Sharpened contribution (one sentence)

> We show whether a ≤60M-parameter, dot-aware render-and-verify model — trained only on
> synthetic Sindhi with native-reviewed dot-confusion negatives — detects and localises dot
> errors in real Sindhi OCR better than OCR confidence, text-only checking, multi-engine voting,
> and a general 2B verifier (MiniCPM-OCR Jev), with the gain concentrated on real-word errors.

**Verdict:** the generic idea is done (Jev, OCR-EDR). The project is still viable as a
**script-specific, low-resource, small-model** study. Smallest pivot already applied above:
frame it as "do render-and-verify OCR checkers work for dot-differentiated scripts?", with
Jev (zero-shot) and OCR-EDR as the explicit points of comparison (OCR-EDR has no released code
found, so cite it, don't run it). If Jev zero-shot already scores high on Sindhi dot errors,
the paper becomes a benchmark + analysis paper; still publishable at LoResLM/CHiPSAL.

## 5. Risks spotted now

- Jev doc: synthetic-only negatives fail on real errors. Same risk for us → Phase 2 real set is
  essential; we must keep some real OCR errors for the final eval only.
- Jev 2B at fp16 ≈ 4–5 GB: fits a T4. GGUF/llama.cpp also available. Budget it in the 8 h baseline slot.
- Tesseract `snd` exists in tessdata_best (per your own DATA_SOURCES.md; re-verify at Phase 2).
- Windows Pillow has no raqm → all rendering must run on Kaggle/WSL (you already found this).

## 6. Resources found in C:\Projects you can reuse

| Resource | Path | Use here | License note |
|---|---|---|---|
| Sindhi Wikipedia 2023-11-01, 66,141 lines (+5k/20k pilots) | `Morph/morphtok-sd-ur/data/processed/sd_wiki20231101_pilot66141.txt` | Text for synthetic pairs; split by sentence | CC BY-SA 3.0 / GFDL (dataset card; corrected in Phase 1) |
| FineWeb-2 Sindhi sample, 35,550 lines | `Morph/.../processed/sd_fineweb2test_pilot35550.txt` | Extra text / LM | ODC-BY (check), may contain Urdu |
| `sindhinorm` (your PyPI package) + native-reviewed 24 "never merge" pairs | `Morph/morphtok-sd-ur/packages/sindhinorm`, `docs/sdnorm.md` | Normalise text; **seed for the ه/ھ/ہ, ي/ى/ی, ک/ڪ/ك part of the confusion set** | MIT code, CC BY-SA word list |
| Synthetic line renderer (raqm, RTL, `language="snd"`) | `Sindhi-Benchmark/ocr/render/render_lines.py` | Base of the scan + candidate renderer | yours |
| Scan degradation (blur, ink, texture, …) + real-vs-synth compare sheet | `Sindhi-Benchmark/ocr/render/degrade.py` | "Scan" augmentation | yours |
| Tesseract/VLM/Kraken baseline eval, LM correction, CER metrics | `Sindhi-Benchmark/ocr/baseline_eval.py`, `ocr/lm_correct.py`, `sindhi_voice_ocr/metrics.py` | Baselines (a), (c) | yours |
| Corpus cleaner with Urdu filter; KenLM build | `Sindhi-Benchmark/lm/clean_corpus.py`, `lm/build_lm.sh` | Text-only baseline (b) | yours |
| Data-source + license log template | `Sindhi-Benchmark/docs/DATA_SOURCES.md`, `logs/source_log.csv` | License tracking | — |
| Spell-checker: dictionary.json (37 MB), frequencies, bigrams | `core-data/Sindhi-Spell-Checker.zip` | Dictionary plausibility for baseline (b) | **check sources in its `docs/dataset_sources.md`** |
| Alphabet reference | same zip, `data/sindhi_alphabet_ref.json` | Glyph-coverage check list | — |

Not found: any font files (need to download Lateef, Scheherazade New, Noto Naskh/Nastaliq, Amiri),
any real Sindhi scans, any OCR test pages (`Sindhi-Benchmark/eval/ocr_test` is empty).
Not relevant: RHAI, SWAI, VIsta, flood, unlearning, tracewarden, swe-inject, app projects.

## 7. Sources

- https://github.com/OpenBMB/MiniCPM-OCR ; https://huggingface.co/openbmb/MiniCPM-OCR-Jev-gguf ; https://huggingface.co/datasets/openbmb/JevOCRBench
- https://arxiv.org/abs/2609.03445 (OCR-EDR)
- https://patents.google.com/patent/US5889897A/en
- https://arxiv.org/abs/2409.04117 (ConfBERT)
- https://arxiv.org/pdf/2606.18884 ; https://arxiv.org/pdf/2604.12978 ; https://arxiv.org/pdf/2505.13943 (partial only)
- https://www.cs.tau.ac.il/~nachumd/papers/ImageAndText.pdf (partial)

## Update 2026-11-18 (new related work found during Phase 3)

- Xiaomi-OCR-0 Technical Report (arXiv 2609.36136): renders a candidate annotation back and compares it with the
  source region in a verification loop (budget of 3 rounds), routing unresolved regions to humans.
- Multimodal OCR (arXiv 2603.13032): render-based verification inside its data pipeline.
- A 2025 ESWA paper on end-to-end Sindhi text recognition reports a real printed set (RW-Sindhi); availability
  unverified. Proxima AI synthetic Sindhi OCR set (CC BY-NC 4.0).
Effect on the contribution: unchanged; render-and-verify is established in large systems, our claims are the small
dot-level Sindhi checker, the real Sindhi benchmark and the real-word error analysis. Details:
`docs/phase3_plan_and_research.md`.
