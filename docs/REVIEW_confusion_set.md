# How to review the confusion set (about 45–60 min)

You review two things: the letter pairs (CSV) and the rendered pictures (PNG).

## 1. Open the files

- `data/confusion_review.csv`: open in Excel (it is UTF-8 with BOM, so Sindhi shows correctly).
  If letters show as `Ø§`, use Data → From Text/CSV → File origin "65001: Unicode (UTF-8)".
- `data/font_sheet.png`: each font gets 2 rows. Row 1: every confusion group, in the same order as the CSV
  (groups separated by wide gaps). Row 2: a test phrase `ٻار ڀاڄي ڏٺي ڍڳو ڙيڻ ڦٽو ڇڏيو ڃڱڻ ڪتاب ۽ گهر`.
  Fonts top to bottom: Amiri, Harmattan, Lateef, Noto Kufi, Noto Naskh, Noto Sans, Scheherazade New.
- `data/synth/test_font_neg_sheet.png`: real examples. Red tag = wrong candidate.
  Left = degraded "scan" of the TRUE text; right = clean candidate, red shade = the changed letter.

## 2. Fill the CSV (one row = one pair of letters in a group)

| Column | What to write |
|---|---|
| `is_valid_pair` | **Y** if the two letters really share a shape and differ only as `kind` says (dot / mark / digraph / encoding). **N** if it is a wrong pair. |
| `ocr_confusable` | **H** / **M** / **L**: how likely a printed-text OCR mixes them up, from your experience reading Sindhi print. |
| `dots_correct` | **Y** if both `dots_a` and `dots_b` descriptions are right; else **N** and write the right one in `comment`. Check ٽ, ڏ, ڙ, ڱ especially. |
| `comment` | Anything else, e.g. "only confusable at word end", "these are also confused with X". |

Missing pairs: add a new row at the bottom with `group_id = NEW` and fill `letter_a`, `letter_b`, `comment`
(why they are confusable). Leave the `code_*` columns empty; I fill them.

Save as CSV UTF-8 (File → Save As → "CSV UTF-8 (Comma delimited)"), same name.

## 3. Look at the pictures

In `font_sheet.png`, write in a text file (or tell me) any font where a letter is drawn **wrongly or unusually
for Sindhi** (e.g. dots in the wrong arrangement, ڪ drawn like ک). Name the font and letter.
In `test_font_neg_sheet.png`, tell me if any "scan" is unreadable or if a dot looks lost in the scan.

## 4. Answer the three decisions (one line each)

1. Should ه/ھ/ہ, ي/ی, ڪ/ک/ك variants count as **errors** when an OCR outputs the "wrong" one? (now: excluded
   from the main metric and reported separately)
2. Held-out scan fonts = Harmattan + Noto Kufi. OK, or another pair?
3. Crops: 1–4 words now. Also full lines?

When done, tell me "review done". I then fold your answers into `data/confusion_set.json` (version
draft-1 → v1), rebuild the fixed sets, and log the changes.
