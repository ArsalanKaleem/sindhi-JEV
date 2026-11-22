"""Make OCR-like candidates from true text, with per-character mismatch labels.

Labels are per code point of the CANDIDATE (that is what the verifier sees):
  substitution / confusion swap -> every code point of the new unit = 1
  insertion                     -> the inserted code point = 1
  deletion                      -> the code point right after the gap = 1
                                   (or right before it if the gap is at the end)
Units keep the digraphs جه / گه together (alphabet.units).
"""
import json
import random
from pathlib import Path

from .alphabet import DIACRITICS, EXTRA_CHARS, SINDHI_ALPHABET, SINDHI_LETTERS, units

NON_JOINERS = set("اآأدڌڏڊڍذرڙزوؤء۽")
JOINABLE_NEXT = (set(SINDHI_LETTERS) | set(EXTRA_CHARS)) - {"۽", "ء"}
ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MIX = {"positive": 0.5, "confusion": 0.3, "random_sub": 0.1, "deletion": 0.05, "insertion": 0.05}


def load_groups(path=None, kinds=("dot", "mark", "digraph")):
    data = json.load(open(path or ROOT / "data" / "confusion_set.json", encoding="utf-8"))
    return [g for g in data["groups"] if g["kind"] in kinds and g["weight"] > 0]


def _joins_next(us, i):
    """True if unit i is drawn in a joined (initial/medial) form."""
    if us[i][-1] in NON_JOINERS:
        return False
    j = i + 1
    while j < len(us) and us[j] in DIACRITICS:  # harakat sit between letters
        j += 1
    return j < len(us) and us[j][:1] in JOINABLE_NEXT


def swap_options(us, groups):
    """All (unit_index, group_id, new_unit, weight) confusion swaps for a text."""
    opts = []
    for i, u in enumerate(us):
        for g in groups:
            partners = group_partners(g).get(u)
            if not partners:
                continue
            if g["positions"] == "joined" and not _joins_next(us, i):
                continue
            for o, mult in partners:
                opts.append((i, g["id"], o, g["weight"] * mult / len(partners)))
    return opts


LEVEL_MULT_DEFAULT = {"H": 1.0, "M": 0.5, "L": 0.2}


def group_partners(g):
    """unit -> [(partner, level multiplier)]. v1 uses the reviewed 'pairs'; draft groups use all members."""
    out = {}
    if "pairs" in g:
        for p in g["pairs"]:
            m = LEVEL_MULT_DEFAULT[p["level"]]
            out.setdefault(p["a"], []).append((p["b"], m))
            out.setdefault(p["b"], []).append((p["a"], m))
    else:
        cs = [m["c"] for m in g["members"]]
        out = {a: [(b, 1.0) for b in cs if b != a] for a in cs}
    return out


def _assemble(us, changed):
    """Join units into a string; label code points of units whose index is in `changed`."""
    cand, labels = "", []
    for i, u in enumerate(us):
        cand += u
        labels += [1 if i in changed else 0] * len(u)
    return cand, labels


def realword_options(us, groups, lexicon):
    """Confusion swaps whose changed word is itself a dictionary word (a 'real-word' error: invisible to a
    spell checker, so only the image can reveal it)."""
    out = []
    for i, gid, new, w in swap_options(us, groups):
        a = i
        while a > 0 and us[a - 1] != " ":
            a -= 1
        e = i
        while e < len(us) and us[e] != " ":
            e += 1
        word = "".join(us[a:i]) + new + "".join(us[i + 1:e])
        if word in lexicon:
            out.append((i, gid, new, w))
    return out


def make_candidate(text, groups, rng=None, mode=None, mix=None, max_swaps=2, lexicon=None):
    """Return dict(cand, labels, mode, edits). mode=None samples from `mix`.
    mode 'realword' (needs lexicon) returns mode 'none' when the text has no real-word swap (caller resamples)."""
    rng = rng or random.Random()
    mix = mix or DEFAULT_MIX
    if mode is None:
        mode = rng.choices(list(mix), weights=list(mix.values()))[0]
    us = units(text)
    letters = [i for i, u in enumerate(us) if u in SINDHI_ALPHABET]
    edits = []

    if mode == "realword":
        opts = realword_options(us, groups, lexicon or set())
        if not opts:
            return {"cand": text, "labels": [0] * len(text), "mode": "none", "edits": []}
        i, gid, new, _w = rng.choices(opts, weights=[o[3] for o in opts])[0]
        edits.append({"op": "swap", "unit": i, "group": gid, "old": us[i], "new": new})
        us[i] = new
        cand, labels = _assemble(us, {i})
        return {"cand": cand, "labels": labels, "mode": mode, "edits": edits}

    if mode == "confusion":
        opts = swap_options(us, groups)
        if not opts:
            mode = "random_sub"  # no confusable letter in this text
        else:
            k = 1 if rng.random() < 0.7 or len(opts) < 2 else max_swaps
            changed = set()
            for _ in range(k):
                avail = [o for o in opts if o[0] not in changed]
                if not avail:
                    break
                i, gid, new, _w = rng.choices(avail, weights=[o[3] for o in avail])[0]
                edits.append({"op": "swap", "unit": i, "group": gid, "old": us[i], "new": new})
                us[i] = new
                changed.add(i)
            cand, labels = _assemble(us, changed)
            return {"cand": cand, "labels": labels, "mode": mode, "edits": edits}

    if mode == "random_sub" and letters:
        i = rng.choice(letters)
        new = rng.choice([a for a in SINDHI_ALPHABET if a != us[i]])
        edits.append({"op": "sub", "unit": i, "old": us[i], "new": new})
        us[i] = new
        cand, labels = _assemble(us, {i})
        return {"cand": cand, "labels": labels, "mode": mode, "edits": edits}

    if mode == "deletion" and len(letters) > 1:
        i = rng.choice(letters)
        edits.append({"op": "del", "unit": i, "old": us[i]})
        del us[i]
        cand, labels = _assemble(us, set())
        # mark the code point right after the gap, else right before it
        pos = sum(len(u) for u in us[:i])
        labels[pos if pos < len(labels) else pos - 1] = 1
        return {"cand": cand, "labels": labels, "mode": mode, "edits": edits}

    if mode == "insertion" and letters:
        i = rng.choice(letters)
        new = rng.choice(SINDHI_ALPHABET)
        edits.append({"op": "ins", "unit": i, "new": new})
        us.insert(i, new)
        cand, labels = _assemble(us, {i})
        return {"cand": cand, "labels": labels, "mode": mode, "edits": edits}

    return {"cand": text, "labels": [0] * len(text), "mode": "positive", "edits": []}
