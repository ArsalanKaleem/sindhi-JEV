import json
import subprocess
import sys
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "real"))

from sindhijev import realset as R  # noqa: E402
from sindhijev.align import derive_labels  # noqa: E402


def test_dot_swap_is_one_substitution_with_group():
    d = derive_labels("ٻار", "بار")
    assert d["char_labels"] == [1, 0, 0]
    assert d["errors"][0]["group"] == "beh"


def test_insertion_and_deletion_rules_match_synthetic():
    d = derive_labels("ابت", "اٻبت")  # inserted ٻ
    assert d["char_labels"] == [0, 1, 0, 0]
    d = derive_labels("ابت", "ات")  # deleted ب -> mark the char after the gap (ت)
    assert d["char_labels"] == [0, 1]
    d = derive_labels("ابت", "اب")  # deleted at the end -> mark last char
    assert d["char_labels"] == [0, 1]


def test_digraph_tagged():
    d = derive_labels("جهڙو", "جڙو")
    assert d["errors"][0]["group"] == "jeem_digraph"


def test_correct_candidate():
    d = derive_labels("سنڌ", "سنڌ")
    assert d["seq_label"] == 0 and d["char_labels"] == [0, 0, 0]


def _gold(cid, text, status="ok", amb="", ovr="", p="1"):
    return {"crop_id": cid, "pass": p, "gold_text": text, "status": status, "ambiguous_chars": amb,
            "override_json": ovr, "seconds": "10", "labeled_at": ""}


CANDS = [{"crop_id": "a", "engine": "e1", "cand_text": "بار", "seq_conf": "0.9"},
         {"crop_id": "a", "engine": "e2", "cand_text": "ٻار", "seq_conf": "0.8"},
         {"crop_id": "b", "engine": "e1", "cand_text": "سنڌ", "seq_conf": "0.7"},
         {"crop_id": "c", "engine": "e1", "cand_text": "x", "seq_conf": ""}]


def test_derive_all_overrides_ambiguous_and_status():
    gold = [_gold("a", "ٻار", amb="0", ovr=json.dumps({"e2": [2]})), _gold("b", "سنڌ"), _gold("c", "?", status="illegible")]
    rows = {(r["crop_id"], r["engine"]): r for r in R.derive_all(CANDS, gold)}
    assert ("c", "e1") not in rows  # illegible crops dropped
    assert rows[("a", "e1")]["seq_label"] == 1
    assert json.loads(rows[("a", "e2")]["char_labels_json"]) == [0, 0, 1]  # override flipped pos 2
    assert json.loads(rows[("a", "e1")]["ambiguous_cand_json"]) == [0]
    assert rows[("b", "e1")]["seq_label"] == 0


def test_agreement_perfect_and_kappa():
    p1 = [_gold("a", "ٻار"), _gold("b", "سنڌ")]
    p2 = [_gold("a", "ٻار", p="2"), _gold("b", "سنڌ", p="2")]
    rep = R.agreement(CANDS, p1, p2)
    assert rep["gold_exact_match"] == 1.0 and rep["seq_label_kappa"] == 1.0
    assert R.cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == -1.0


def test_load_real_requires_final(tmp_path):
    with pytest.raises(PermissionError):
        R.load_real()


def test_hocr_parser():
    from run_ocr import parse_hocr
    hocr = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "x">
<html xmlns="http://www.w3.org/1999/xhtml"><body>
<span class='ocr_line' title='bbox 0 0 10 10'>
<span class='ocrx_word' title='bbox 0 0 5 5; x_wconf 91'><span class='ocrx_cinfo' title='x_bboxes 0 0 1 1; x_conf 99.5'>ٻ</span><span class='ocrx_cinfo' title='x_bboxes 0 0 1 1; x_conf 50'>ا</span></span>
<span class='ocrx_word' title='bbox 5 0 9 5; x_wconf 80'>ر</span>
</span></body></html>"""
    text, wc, cc = parse_hocr(hocr)
    assert text == "ٻا ر" and wc == [0.91, 0.80]
    assert cc == []  # 2nd word has no char info -> no partial char confs


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_labeler_js_matches_python():
    html = (ROOT / "tools" / "labeler.html").read_text(encoding="utf-8")
    start = html.index("function deriveLabels")
    end = html.index("const norm")
    js = html[start:end]
    pairs = [("ٻار", "بار"), ("ابت", "اٻبت"), ("ابت", "ات"), ("ابت", "اب"), ("جهڙو", "جڙو"),
             ("سنڌ جي تاريخ", "سند جي تارخ"), ("", "اب"), ("اب", "")]
    script = js + "\nconst P=" + json.dumps(pairs, ensure_ascii=False) + \
        ";console.log(JSON.stringify(P.map(([g,c])=>deriveLabels(g,c).lab)));"
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, encoding="utf-8", check=True).stdout
    assert json.loads(out) == [derive_labels(g, c)["char_labels"] for g, c in pairs]


def test_rejected_pair_tagged_other_but_encoding_kept():
    assert derive_labels("فقير", "ققير")["errors"][0]["group"] == "other"  # ف/ق rejected in review
    assert derive_labels("ڪتاب", "كتاب")["errors"][0]["group"] == "kaf_enc"  # encoding variant still monitored


def test_latin_punctuation_equals_arabic_and_overrides_can_be_ignored():
    from sindhijev.realset import derive_all, norm
    assert norm("سڪون, برائي?") == "سڪون، برائي؟"
    cands = [{"crop_id": "c", "engine": "e", "cand_text": "بار،", "seq_conf": "", "char_conf_json": ""}]
    gold = [{"crop_id": "c", "pass": "1", "gold_text": "بار,", "status": "ok", "ambiguous_chars": "",
             "override_json": '{"e": [0]}'}]
    assert derive_all(cands, gold)[0]["overridden"] == "Y"
    r = derive_all(cands, gold, ignore_overrides=True)[0]
    assert r["overridden"] == "N" and r["seq_label"] in (0, "0")


def test_fold_encoding_variants_and_bidi_marks():
    from sindhijev.realset import fold_encoding
    assert fold_encoding("ڊگري‏ ۽ ڊیوٽي ھا") == fold_encoding("ڊگري ۽ ڊيوٽي ها") == "ڊگري ۽ ڊيوٽي ها"
    assert fold_encoding("ٿو") != fold_encoding("ڏو")  # dot differences are never folded
