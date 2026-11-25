# sindhijev

Check Sindhi OCR text against the scanned word and flag dot errors letter by letter.

Sindhi letters differ by dots (ب ٻ ڀ پ, ن ڻ, ج ڄ, د ڏ ڌ). OCR engines often get the shape right and the dots wrong,
and many such mistakes are still real words. `sindhijev` looks at the word crop and the OCR text together, scores how
likely the text is wrong, and flags the suspicious letters. It is a 7M-parameter model run with ONNX Runtime on CPU
(about 20 ms per crop); the model file (~30 MB) is downloaded once on first use and checked by SHA-256.

```
pip install sindhijev

sindhijev check word.png "ڪلوواٽ اهي"       # p_wrong=1.000, suspicious letter: ا
sindhijev repair word.png "ڪلوواٽ اهي"      # ڪلوواٽ آهي
sindhijev pick word.png --cand "..." --cand "..."  [--tesseract]
```

```python
from sindhijev.tool import Checker
r = Checker().check("word.png", "ڪلوواٽ اهي")
print(r["p_wrong"], r["suspicious"])
```

**Scope.** Built for crops of 1 to 4 printed Sindhi words. On a frozen, hand-labelled set of real scans it ranks
wrong OCR outputs above correct ones with AUROC 0.884 (0.935 on dot errors); on full lines it does not work.

- Documentation: https://arsalankaleem.me/sindhi-JEV/
- Live demo: https://huggingface.co/spaces/ArsalanKaleem/sindhijev
- Source, data and results: https://github.com/ArsalanKaleem/sindhi-JEV
- Licence: Apache-2.0 (bundled Noto Naskh Arabic font: SIL Open Font License)
