"""The CPU tool must feed the model exactly what training/evaluation feeds it (no weights needed)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("torch")


class FakeSess:
    def __init__(self):
        self.feeds = []

    def run(self, _, feeds):
        self.feeds.append(feeds)
        B, L = feeds["ids"].shape
        return np.zeros(B, np.float32), np.zeros((B, L), np.float32)


def make_checker():
    from sindhijev.render import Renderer
    from sindhijev.tool import Checker
    ck = Checker.__new__(Checker)
    ck.sess, ck.H, ck.max_w, ck.max_chars, ck.temperature, ck.char_thr = FakeSess(), 48, 1024, 96, 1.0, 0.5
    ck.r = Renderer(ROOT / "src" / "sindhijev" / "assets" / "NotoNaskhArabic.ttf", size=48)
    return ck


def test_tool_inputs_match_collate():
    from sindhijev.data import collate, to_example
    ck = make_checker()
    scan = (np.random.default_rng(0).random((60, 300)) * 255).astype(np.uint8)
    texts = ["سنڌ جي تاريخ", "ڪتاب", "جهڙو ٻار ڏٺو"]
    ck.check_many(scan, texts)
    assert len(ck.sess.feeds) == len(texts)  # one forward pass per candidate (no batch padding)
    for got, t in zip(ck.sess.feeds, texts):
        img, cx = ck.r.render(t)
        ref = collate([to_example(scan, img, t, cx, 0, [0] * len(t), {"img_height": 48, "max_width": 1024,
                                                                      "max_chars": 96})])
        for k in ("scan", "scan_w", "cand", "cand_w", "ids", "char_x"):
            np.testing.assert_allclose(got[k], ref[k].numpy(), atol=1e-6, err_msg=k)


def test_pick_orders_by_p_wrong_and_flags_letters():
    ck = make_checker()

    from sindhijev.data import encode
    beeh = encode("ٻ")[0]

    def run(feeds):  # "ٻار" looks wrong (logit +2), "بار" looks right (-2); letter 1 is always suspicious
        B, L = feeds["ids"].shape
        seq = np.where((feeds["ids"] == beeh).any(1), 2.0, -2.0).astype(np.float32)
        return seq, np.tile(np.array([[-5.0, 5.0] + [-5.0] * (L - 2)]), (B, 1))
    ck.sess.run = lambda _, feeds: run(feeds)
    ranked = ck.pick(np.full((40, 120), 255, np.uint8), ["ٻار", "بار"])
    assert [r["text"] for r in ranked] == ["بار", "ٻار"] and ranked[0]["p_wrong"] < 0.5
    assert ranked[0]["suspicious"] == [1]


def test_packaged_confusion_set_matches_data():
    assert (ROOT / "src" / "sindhijev" / "assets" / "confusion_set.json").read_bytes() == \
        (ROOT / "data" / "confusion_set.json").read_bytes()


def test_packaged_model_card_has_calibration():
    import json
    card = json.loads((ROOT / "src" / "sindhijev" / "assets" / "sindhijev_v2.json").read_text(encoding="utf-8"))
    assert 0.5 < card["temperature"] < 3 and 0 < card["char_threshold"] < 1 and card["model_cfg"]["img_height"] == 48
