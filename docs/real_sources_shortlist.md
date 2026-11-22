# archive.org shortlist for the real test set (2026-11-17), awaiting approval

Found by archive.org metadata search (`language:Sindhi OR snd`, 1,500 items, Sinhala and
Devanagari-script items removed). Only metadata has been read; **nothing is downloaded**.
"Scan?" comes from the PDF type: *Image Container PDF* = page images (scan);
*Text PDF* = scan with a text layer **or** a retyped/born-digital edition (unknown until opened).

Rights tiers:
- **PD**: public domain with evidence (date + author death year, or a "not in copyright" mark from the scanner).
- **PD?**: very likely PD (published 80+ years ago) but the author's death year is not verified.
- **Lic**: an explicit CC license on the item.
- **Eval-only**: probably in copyright. Used for evaluation only; we publish archive.org ID + page + bbox + gold text, never the images.

| # | archive.org ID | Title (year) | Scan? | PDF file (size) | Rights | Why |
|---|---|---|---|---|---|---|
| 1 | `grammarsindhila03trumgoog` | Trumpp, *Grammar of the Sindhi Language* (1872) | scan (Google) | `grammarsindhila03trumgoog.pdf` (16.5 MB, 616 p) | **PD**: Google "NOT_IN_COPYRIGHT", author d. 1885 | 19th-c. movable type, Sindhi words in Arabic script inside English text |
| 2 | `sindi_bcp1873` | Sindhi Book of Common Prayer (1873) | scan, 150 ppi | `sindi_bcp1873.pdf` (3.0 MB, 35 p) | **PD** content (1873); item marked CC BY-NC-ND 4.0 | old type, low resolution |
| 3 | `san_0460` | تنبيه الجاهلين (1872) | Text PDF (?) | `تنبيه_الجاهلين-…-1872ع.pdf` (1.6 MB) | **PD** (1872) | lithograph if it is a scan, which makes for hard crops |
| 4 | `1923_20260819_202608` | چچ نامو, Mirza Qalich Beg (1923) | scan, 264 ppi | `چچ نامو (تاريخ 1923) مرزا قليچ بيگ.pdf` (24.4 MB) | **PD**: author d. 1929 | classic early-20th-c. Sindhi print |
| 5 | `san_0046` | ماهوار سنڌو, April issue (1938) | scan, 300 ppi | `ماهوار_سنڌو-(1938-04)-اپريل-صاف.pdf` (7.2 MB) | **PD?** (1938, editor B. W. Rajpal) | magazine layout |
| 6 | `san_0103` | ڪلمة الحق, Maulvi Mohammad Sadiq (1934) | Text PDF (?) | `ڪلمة_الحق-…-1934-صاف.pdf` (1.9 MB) | **PD?** | 1930s print |
| 7 | `RaddMizan-ul-insafsindhi1936` | Radd Mizan-ul-Insaf, M. B. Wasif (1936) | scan, 600 ppi | `00391_RADD-MEEZAN-UL-INSAF.pdf` (50.3 MB) | **Lic** CC BY-NC-ND 3.0 (+ PD? content) | high-resolution scan |
| 8 | `5-1-1956` | ٽماھي مھراڻ vol. 5 no. 1 (1956) | scan, 72 ppi | `ٽماھي مھراڻ (جلد 5 نمبر 1 سال 1956ع) ….pdf` (38.6 MB) | **Eval-only** | 1950s journal, low resolution |
| 9 | `san_0850` | ٽماهي مهراڻ vol. 8 no. 4 (1959) | scan, 300 ppi | `ٽماهي_مهراڻ_جلد_8_نمبر_4-…-1959-صاف.pdf` (8.9 MB) | **Eval-only** | same journal, better scan |
| 10 | `san_0836` | اردو سنڌي لغت, Dr N. B. Baloch (1960) | scan, 300 ppi | `اردو_سنڌي_لغت-…-صاف.pdf` (14.9 MB) | **Eval-only** | dictionary: dense, many dot-heavy words |
| 11 | `soorih-sindhi-adab_20250312_1710` | هفتا، ڏينھن، راتيون ۽ مهينا, Dr N. B. Baloch (1961) | scan, 72 ppi | `هفتا، ڏينھن، راتيون ۽ مهينا.pdf` (97.9 MB) | **Eval-only** | low resolution 1960s print |
| 12 | `2002_20231018` | سنڌ ھڪ نظر ۾, A. M. Dharejo (2002) | Text PDF (?) | `سنڌ_ھڪ_نظر_۾-…-2002-صاف.pdf` (3.6 MB) | uploader's PD mark, unverified, so **Eval-only** | modern computer typesetting |
| 13 | `1512272497_202106` | اخلاقيات class 4 textbook (2020) | Text PDF (?) | `1512272497.pdf` (27.2 MB) | uploader's PD mark, unverified (textbook board), so **Eval-only** | modern school print |
| 14 | `dil-degree-duty-.pdf-1` | دل ڊگري ڊيوٽي (2025) | Text PDF, likely born-digital | `dil degree duty دل ڊگري ڊيوٽي.pdf1.pdf` (0.8 MB) | **Lic** CC BY-NC 4.0 | modern digital font (render it, then degrade by printing or phone-photo) |

Dropped: `SND1890_DBS_HS` (its PDF is named "Shona-John", i.e. it doesn't match the title);
all `dli.language.*` items (Sindhi in **Devanagari**).

## Proposed download (needs your OK)

- **Option A, page sample (recommended):** for each item, fetch only 8 random page images
  through archive.org's per-page image URLs (`https://archive.org/download/<id>/page/n<k>.jpg`,
  if the item has them; otherwise fall back to B for that item). About 14 × 8 × ~0.3 MB ≈ **35 MB**.
- **Option B, full PDFs:** about **320 MB** in total (largest: #11 at 97.9 MB, #7 at 50.3 MB).

Files go to `data/real/pages/` as `<source_id>__p<k>.jpg`, plus one row each in `data/real/sources.csv`.
Downloaded files are treated as untrusted data (images only, no code is run from them).

## Still needed beyond archive.org

- Newspapers (e-paper screenshots or scans you have): eval-only.
- Your handwriting: 50–100 crops (write new sentences).
- Optionally, phone photos of Sindhi print you own (adds camera noise, which no item above has).

Risks to keep in mind: the 1872–73 items may use pre-standard letter forms or dot
conventions, so gold labelling there is harder; keep them as their own stratum
("historical") rather than mixing them with modern print.
