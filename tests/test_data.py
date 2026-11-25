import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sindhijev.alphabet import SINDHI_ALPHABET, check_names, units  # noqa: E402
from sindhijev.corrupt import load_groups, make_candidate, swap_options  # noqa: E402
from sindhijev.render import Renderer  # noqa: E402

GROUPS = load_groups()
FONT = ROOT / "fonts" / "NotoNaskhArabic.ttf"


def test_alphabet_codepoints():
    assert check_names() == []
    assert len(SINDHI_ALPHABET) == 52
    assert len(set(SINDHI_ALPHABET)) == 52


def test_units_keep_digraphs():
    assert units("گهر") == ["گه", "ر"]
    assert units("جهڙو") == ["جه", "ڙ", "و"]


def test_confusion_groups_members_unique_and_valid():
    data = json.load(open(ROOT / "data" / "confusion_set.json", encoding="utf-8"))
    for g in data["groups"]:
        cs = [m["c"] for m in g["members"]]
        assert len(cs) == len(set(cs)), g["id"]


@pytest.mark.parametrize("mode", ["positive", "confusion", "random_sub", "deletion", "insertion"])
def test_labels_align_with_candidate(mode):
    rng = random.Random(0)
    for text in ["ٻار ڀاڄي ڏٺي", "ڪتاب ۽ گهر", "سنڌ جي تاريخ"]:
        for _ in range(50):
            c = make_candidate(text, GROUPS, rng, mode=mode)
            assert len(c["labels"]) == len(c["cand"])
            if c["mode"] == "positive":
                assert c["cand"] == text and not any(c["labels"])
            else:
                assert c["cand"] != text or c["mode"] == "invisible"
                assert 1 <= sum(c["labels"]) <= 4


def test_confusion_swap_stays_in_group():
    rng = random.Random(1)
    for _ in range(200):
        c = make_candidate("ٻار ڀاڄي ڏٺي ڍڳو", GROUPS, rng, mode="confusion")
        for e in c["edits"]:
            g = next(g for g in GROUPS if g["id"] == e["group"])
            members = {m["c"] for m in g["members"]}
            assert e["old"] in members and e["new"] in members and e["old"] != e["new"]


def test_joined_group_only_in_joined_position():
    # final ب in "اب" is not joined to a next letter -> beh_joined must not apply
    opts = swap_options(units("اب"), GROUPS)
    assert not any(gid == "beh_joined" for _, gid, _, _ in opts)
    opts = swap_options(units("بت"), GROUPS)
    assert any(gid == "beh_joined" and i == 0 for i, gid, _, _ in opts)


def test_deletion_marks_char_after_gap():
    rng = random.Random(3)
    c = make_candidate("ابت", GROUPS, rng, mode="deletion")
    i = c["edits"][0]["unit"]
    pos = min(i, len(c["cand"]) - 1)
    assert c["labels"][pos] == 1


def test_render_rtl_and_dot_letters_differ():
    r = Renderer(FONT, size=32)
    img, cx = r.render("گهر ٻار", pad=1)
    assert cx[0][0] > cx[-1][0]  # first logical char is on the right
    a, _ = r.render("ب")
    b, _ = r.render("ٻ")
    assert a.shape != b.shape or not np.array_equal(a, b)


def test_reviewed_excluded_pairs_never_generated():
    data = json.load(open(ROOT / "data" / "confusion_set.json", encoding="utf-8"))
    assert data["version"] == "v1"
    banned = {(p["a"], p["b"]) for g in data["groups"] for p in g.get("excluded_pairs", [])}
    banned |= {(b, a) for a, b in banned}
    rng = random.Random(5)
    for text in ["بني تين ڦڦڙ فقير", "جهڙو ڄاڻ ڻبين"]:
        for i, _gid, new, _w in swap_options(units(text), GROUPS):
            assert (units(text)[i], new) not in banned
        for _ in range(200):
            for e in make_candidate(text, GROUPS, rng, mode="confusion")["edits"]:
                assert (e["old"], e["new"]) not in banned


def test_fill_ranges_borrows_neighbour_range():
    from sindhijev.data import fill_ranges
    assert fill_ranges([(5, 9), None, (1, 4)]) == [(5, 9), (5, 9), (1, 4)]
    assert fill_ranges([None, (1, 4)]) == [(1, 4), (1, 4)]
    assert fill_ranges([None]) == [(0, 0)]
