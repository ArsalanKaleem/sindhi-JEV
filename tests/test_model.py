"""Phase 3 code: metrics vs sklearn, model shapes for every ablation, batching, real-word corruption."""
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
torch = pytest.importorskip("torch")
from sindhijev.alphabet import units  # noqa: E402
from sindhijev.corrupt import load_groups, make_candidate, realword_options  # noqa: E402
from sindhijev.data import Manifest, collate  # noqa: E402
from sindhijev.metrics import auprc, auroc, char_metrics, routing  # noqa: E402
from sindhijev.model import Verifier, n_params  # noqa: E402


def test_metrics_match_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = random.Random(0)
    y = [rng.random() < 0.3 for _ in range(300)]
    s = [round(rng.random() + 0.3 * yy, 2) for yy in y]  # rounding creates ties
    y = [int(v) for v in y]
    assert abs(auroc(y, s) - sk.roc_auc_score(y, s)) < 1e-9
    assert abs(auprc(y, s) - sk.average_precision_score(y, s)) < 1e-9


def test_routing_and_char_metrics():
    r = routing([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2], fracs=(0.5,))["route@50"]
    assert r["recall"] == 1.0 and r["precision"] == 1.0 and r["residual_error"] == 0.0
    recs = [{"seq_y": 1, "char_y": [0, 1, 0], "char_score": [0.1, 0.9, 0.2], "ignore": []},
            {"seq_y": 1, "char_y": [1, 0], "char_score": [0.2, 0.8], "ignore": [1]}]
    m = char_metrics(recs, 0.5)
    assert m["top1_loc"] == 1.0 and m["char_f1"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("flags", [{}, {"use_scan": False}, {"use_render": False}, {"char_align": False},
                                   {"char_head": False}, {"use_scan": False, "use_render": False, "char_align": False}])
def test_model_shapes_all_ablations(flags):
    m = Verifier(flags)
    ds = Manifest(ROOT / "data" / "synth" / "val", m.cfg, limit=4)
    b = collate([ds[i] for i in range(4)])
    out = m(b)
    assert out["seq_logit"].shape == (4,)
    if m.cfg["char_head"]:
        assert out["char_logit"].shape == b["ids"].shape
    loss, _ = m.loss(out, b)
    loss.backward()
    assert torch.isfinite(loss)
    assert n_params(m) < 60e6


def test_char_x_inside_padded_width():
    m = Verifier()
    ds = Manifest(ROOT / "data" / "synth" / "val", m.cfg, limit=8)
    b = collate([ds[i] for i in range(8)])
    cx = b["char_x"][b["char_mask"]]
    frac_w = (b["cand_w"].float() / b["cand"].shape[-1])
    assert (cx >= 0).all() and (cx[:, 1] <= frac_w.max() + 1e-6).all()


def test_realword_swap_makes_dictionary_word():
    groups = load_groups()
    lex = {"بار", "ٻار", "تار"}
    us = units("ٻار")
    opts = realword_options(us, groups, lex)
    assert opts and all("".join(us[:i] + [new] + us[i + 1:]) in lex for i, _, new, _ in opts)
    c = make_candidate("ٻار", groups, random.Random(0), mode="realword", lexicon=lex)
    assert c["cand"] in lex and c["cand"] != "ٻار" and sum(c["labels"]) == 1
    assert make_candidate("ٻار", groups, random.Random(0), mode="realword", lexicon=set())["mode"] == "none"


def test_selection_and_ece():
    from sindhijev.metrics import ece, edit_distance, selection
    assert edit_distance("بار", "ٻار") == 1 and edit_distance("", "ab") == 2
    recs = [{"cluster": "c", "engine": "a", "cand": "بار", "gold": "ٻار", "s": 0.9},
            {"cluster": "c", "engine": "b", "cand": "ٻار", "gold": "ٻار", "s": 0.1}]
    assert selection(recs, "s") == {"cer": 0.0, "exact": 1.0, "crops": 1}
    assert ece([1, 0], [1.0, 0.0]) == 0.0 and abs(ece([1, 1], [0.5, 0.5]) - 0.5) < 1e-9
