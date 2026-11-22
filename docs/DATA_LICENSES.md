# Data and asset licenses

| Asset | Source | License | Notes |
|---|---|---|---|
| Sindhi Wikipedia text, dump 2023-11-01 | HF `wikimedia/wikipedia` config `20231101.sd`, via `Morph/morphtok-sd-ur/data/processed/sd_wiki20231101_pilot66141.txt` (SHA-256 in its manifest) | CC BY-SA 3.0 / GFDL (as stated on the dataset card) | Derived text and synthetic images that show it must keep attribution + share-alike. Commercial use allowed under those terms. |
| Fonts: Amiri, Harmattan, Lateef, Noto Kufi Arabic, Noto Naskh Arabic, Noto Sans Arabic, Scheherazade New | `github.com/google/fonts` (`ofl/…`), downloaded 2026-11-17 | SIL OFL 1.1 (license text next to each font in `fonts/`) | Rendered images are not "Font Software", so OFL places no limits on the synthetic dataset. |
| Fonts not used: Noto Nastaliq Urdu, Markazi Text, Vazirmatn | same | OFL | Fail Sindhi coverage (`data/font_coverage.md`). |
| `sindhinorm` normaliser | own package (PyPI) | MIT code, CC BY-SA 4.0 word list | |
| Real scans (Phase 2) | — | — | To be logged per item before use. |

Not used (yet): FineWeb-2 Sindhi (ODC-BY, contains Urdu lines), Sindhi-Spell-Checker
dictionary (check its `docs/dataset_sources.md` before using it in the text-only baseline).
