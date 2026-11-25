# `sindhijev`: check Sindhi OCR text against the scan (CPU)

A 7M-parameter model (v2) exported to ONNX (~30 MB). No GPU and no PyTorch needed.

```
pip install sindhijev          # the model downloads once from the v0.1.0 release (SHA-256 checked)

sindhijev check word.png "سنڌ جي"        # probability the text is wrong + suspicious letters
sindhijev repair word.png "سنڌ جي"       # try dot fixes; keep one only if it clearly wins
sindhijev pick word.png --cand "..." --cand "..."  [--tesseract]   # best of several OCR outputs
```

Python: `from sindhijev.tool import Checker; Checker().check("word.png", "text")`.

**Docs site and playground (local only):** `python site_src/build.py`, then `python tools/demo/server.py`, and open http://localhost:8770 (playground: /playground.html). Publishing steps: `docs/publishing.md`. Upload or paste a word crop (or pick a synthetic sample), type the OCR text, and use Check / Repair / Compare. Nothing leaves the computer; the samples are synthetic (the real test set is never served).

## What the numbers mean (from the frozen real test set, `docs/final_real_results.md`)
- **Use it on word crops (1–4 words).** On real word crops it ranks wrong OCR outputs above correct ones with
  AUROC 0.88–0.90 (same as the 2B MiniCPM-OCR Jev, better than OCR confidence, voting and a text-only checker), and
  0.94 on dot errors.
- **Do not use it on full lines.** On real full lines it is at chance (0.46).
- `p_wrong` is temperature-scaled on synthetic validation data (T = 1.145). On real words its calibration error
  (ECE) is about 0.11: treat it as a ranking score, not an exact probability.
- `suspicious` letters use the threshold chosen on synthetic validation (0.817). On real words the most suspicious
  letter is a wrong one 84% of the time (95% for dot errors); the full flag set is less precise (letter F1 0.47).
- `pick` follows the score. On real words this did **not** beat always using Tesseract `snd`
  (CER 0.215 vs 0.181), so prefer `check` for review triage.
- `repair` (margin 0.6, chosen on synthetic val) is evaluated on synthetic data only (`results/dot_repair.md`): it fixes 98% of real-word dot swaps in training fonts and 87% in unseen fonts, breaking 1–2 correct texts in 2,000. It has not been measured on real scans.

## Known limits
- Each crop/candidate is scored in its own forward pass. In padded batches the score of a crop shifts slightly with
  the other crops in the batch (BatchNorm on padding; `results/batch_effect.json`). All reported numbers were
  computed with batch 64; on the synthetic test sets this changes AUROC by at most 0.0001.
- Encoding variants (ه/ھ/ہ, ي/ی, ڪ/ک/ك) are not counted as errors in the reported metrics.
- Trained on printed Naskh-style fonts; handwriting and Nastaliq are untested.
