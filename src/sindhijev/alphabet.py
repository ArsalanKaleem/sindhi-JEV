"""Sindhi (Perso-Arabic) alphabet and text units.

Code points are checked against `unicodedata` names in tests/test_alphabet.py,
NOT against the spell-checker's sindhi_alphabet_ref.json, which has wrong
entries (Urdu ڈ ڑ, Pashto ډ, a combining hamza for ڻ).

Aspirated digraphs (جه, گه) are written with plain heh U+0647, matching
sindhinorm's canonical form.
"""
import unicodedata

# The 52-letter Sindhi alphabet in dictionary order. Two entries are digraphs.
SINDHI_ALPHABET = [
    "ا", "ب", "ٻ", "ڀ", "ت", "ٿ", "ٽ", "ٺ", "ث", "پ",
    "ج", "ڄ", "جه", "ڃ", "چ", "ڇ", "ح", "خ",
    "د", "ڌ", "ڏ", "ڊ", "ڍ", "ذ", "ر", "ڙ", "ز",
    "س", "ش", "ص", "ض", "ط", "ظ", "ع", "غ",
    "ف", "ڦ", "ق", "ڪ", "ک", "گ", "ڳ", "گه", "ڱ",
    "ل", "م", "ن", "ڻ", "و", "ه", "ء", "ي",
]

# Expected Unicode names: guards against look-alike code points.
EXPECTED_NAMES = {
    "ٻ": "ARABIC LETTER BEEH", "ڀ": "ARABIC LETTER BEHEH",
    "ٿ": "ARABIC LETTER TEHEH", "ٽ": "ARABIC LETTER TEH WITH THREE DOTS ABOVE DOWNWARDS",
    "ٺ": "ARABIC LETTER TTEHEH", "ڄ": "ARABIC LETTER DYEH", "ڃ": "ARABIC LETTER NYEH",
    "ڇ": "ARABIC LETTER TCHEHEH", "ڌ": "ARABIC LETTER DAHAL",
    "ڏ": "ARABIC LETTER DAL WITH THREE DOTS ABOVE DOWNWARDS",
    "ڊ": "ARABIC LETTER DAL WITH DOT BELOW", "ڍ": "ARABIC LETTER DDAHAL",
    "ڙ": "ARABIC LETTER REH WITH FOUR DOTS ABOVE", "ڦ": "ARABIC LETTER PEHEH",
    "ڪ": "ARABIC LETTER SWASH KAF", "ک": "ARABIC LETTER KEHEH",
    "ڳ": "ARABIC LETTER GUEH", "ڱ": "ARABIC LETTER NGOEH", "ڻ": "ARABIC LETTER RNOON",
}

# Other characters common in Sindhi text, plus look-alike variants that OCR
# engines and keyboards produce.
EXTRA_CHARS = ["آ", "ئ", "ؤ", "أ", "۽", "ھ", "ہ", "ى", "ی", "ك", "ة", "ے"]
DIACRITICS = ["َ", "ُ", "ِ", "ّ", "ْ", "ً", "ٌ", "ٍ"]
PUNCT = ["،", "؛", "؟", "۔", ".", ":", "(", ")", "-", "!"]

SINDHI_LETTERS = sorted({c for u in SINDHI_ALPHABET for c in u})
DIGRAPHS = [u for u in SINDHI_ALPHABET if len(u) > 1]

# Characters every font must cover (Phase 1 glyph check).
REQUIRED_CHARS = sorted(set(SINDHI_LETTERS) | set(EXTRA_CHARS) | set(DIACRITICS) | {"،", "؟", "۔"})


def is_sindhi_char(c):
    return c in SINDHI_LETTERS or c in EXTRA_CHARS or c in DIACRITICS


def units(text):
    """Split text into letter units, keeping the digraphs جه and گه together.

    Note: جه / گه can also be a real letter sequence (e.g. across morphemes);
    this split is used only to pick swap positions, labels stay per code point.
    """
    out, i = [], 0
    while i < len(text):
        if text[i:i + 2] in DIGRAPHS:
            out.append(text[i:i + 2])
            i += 2
        else:
            out.append(text[i])
            i += 1
    return out


def check_names():
    """Return a list of (char, expected, actual) mismatches."""
    return [(c, n, unicodedata.name(c, "?")) for c, n in EXPECTED_NAMES.items()
            if unicodedata.name(c, "?") != n]
