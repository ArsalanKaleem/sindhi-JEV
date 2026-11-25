"""Hugging Face Space (Gradio SDK): Sindhi-Jev checker on CPU.

Same Checker as the pip tool (ONNX, one crop per forward pass). Images are processed in memory and not stored.
Locally: python space/app.py  (expects the layout made by space/make_space.py, or run it from dist/hf_space/).
"""
import base64
import html
import json
import os
import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
from sindhijev.tool import Checker  # noqa: E402

CK = Checker(os.environ.get("SINDHIJEV_WEIGHTS", str(HERE / "model" / "sindhijev_v2.onnx")))
REPO = "https://github.com/ArsalanKaleem/sindhi-JEV"

CSS = """
.sj * { box-sizing: border-box; }
.sj { font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
.sj .ar { font-family: "Noto Naskh Arabic", serif; direction: rtl; }
.sj .verdict { display: flex; align-items: center; gap: 16px; padding: 14px 16px; border-radius: 10px; }
.sj .verdict.ok { background: rgba(24,121,78,.12); } .sj .verdict.bad { background: rgba(180,35,24,.12); }
.sj .big { font-size: 28px; font-weight: 700; min-width: 90px; font-variant-numeric: tabular-nums; }
.sj .ok .big { color: #18794e; } .sj .bad .big { color: #c0392b; }
.sj .what { font-weight: 650; } .sj .sub { font-size: 13px; opacity: .75; }
.sj .pair { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 14px 0; }
.sj figure { margin: 0; border: 1px solid rgba(128,128,128,.3); border-radius: 8px; padding: 8px 10px; background: #fff; }
.sj figcaption { font-size: 11.5px; color: #5d6673; margin-bottom: 4px; }
.sj figure img { display: block; max-width: 100%; height: 58px; object-fit: contain; margin-left: auto; }
.sj .chips { display: flex; flex-wrap: wrap; gap: 6px; direction: rtl; margin: 8px 0; }
.sj .chip { min-width: 38px; text-align: center; border-radius: 8px; padding: 3px 6px 1px; border: 1.5px solid transparent; }
.sj .chip .c { font-family: "Noto Naskh Arabic", serif; font-size: 26px; line-height: 1.3; }
.sj .chip .s { font-size: 10.5px; opacity: .7; direction: ltr; }
.sj .chip.sp { min-width: 14px; background: transparent !important; }
.sj .flag { border-color: #c0392b !important; } .sj .changed { border-color: #18794e !important; }
.sj .h { font-weight: 650; margin: 14px 0 4px; } .sj .note { font-size: 13px; opacity: .75; margin-top: 10px; }
.sj .rank { border: 1px solid rgba(128,128,128,.3); border-radius: 10px; padding: 10px 14px; margin-bottom: 10px; }
.sj .rank.best { border-color: #18794e; }
.sj .row { display: flex; align-items: center; gap: 10px; } .sj .txt { font-size: 23px; flex: 1; }
.sj .pill { font-size: 12px; font-weight: 650; padding: 3px 9px; border-radius: 99px; }
.sj .pill.ok { background: rgba(24,121,78,.15); color: #18794e; } .sj .pill.bad { background: rgba(192,57,43,.15); color: #c0392b; }
"""
HEAD = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link href="https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic&display=swap" rel="stylesheet">')


def _png(img):
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def _gray(img):
    if img is None:
        raise gr.Error("Add a word crop first.")
    img = np.asarray(img)
    if img.ndim == 3:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2GRAY)
    if img.shape[0] > 400:
        img = cv2.resize(img, (int(img.shape[1] * 400 / img.shape[0]), 400), interpolation=cv2.INTER_AREA)
    return img.astype(np.uint8)


def _heat(v):
    return f"rgba(214,48,39,{0.06 + 0.5 * min(1.0, max(0.0, v)):.3f})"


def _chips(r, changed=()):
    flag = set(r["suspicious"])
    out = []
    for c in r["chars"]:
        if c["char"] == " ":
            out.append('<span class="chip sp"></span>')
            continue
        cls = "chip" + (" flag" if c["i"] in flag else "") + (" changed" if c["i"] in changed else "")
        out.append(f'<span class="{cls}" style="background:{_heat(c["score"])}" title="letter {c["i"] + 1}: '
                   f'{c["score"]:.3f}"><div class="c">{html.escape(c["char"])}</div><div class="s">{c["score"]:.2f}</div></span>')
    return '<div class="chips">' + "".join(out) + "</div>"


def _verdict(p):
    bad = p >= 0.5
    return (f'<div class="verdict {"bad" if bad else "ok"}"><div class="big">{p * 100:.1f}%</div><div>'
            f'<div class="what">{"Likely wrong" if bad else "Looks correct"}</div>'
            '<div class="sub">chance the text does not match the crop</div></div></div>')


def _pair(scan, text):
    rend, _ = CK.r.render(text or " ")
    return (f'<div class="pair"><figure><figcaption>Scan</figcaption><img src="{_png(scan)}"></figure>'
            f'<figure><figcaption>Text as rendered for the model</figcaption><img src="{_png(rend)}"></figure></div>')


def _wrap(body):
    return f'<div class="sj">{body}</div>'


def check(img, text):
    scan = _gray(img)
    if not (text or "").strip():
        raise gr.Error("Type the OCR text.")
    r = CK.check(scan, text)
    flagged = " ، ".join(html.escape(r["text"][i]) for i in r["suspicious"])
    return _wrap(_verdict(r["p_wrong"]) + _pair(scan, r["text"]) + '<div class="h">Letters, right to left</div>'
                 + _chips(r) + (f'<div class="note">Flagged: <span class="ar" style="font-size:20px">{flagged}</span></div>'
                                if flagged else '<div class="note">No letter above the flag threshold.</div>'))


def repair(img, text):
    scan = _gray(img)
    if not (text or "").strip():
        raise gr.Error("Type the OCR text.")
    fixed, hist = CK.repair(scan, text)
    before, after = CK.check(scan, text), CK.check(scan, fixed)
    changed = {i for i, (a, b) in enumerate(zip(after["text"], before["text"])) if a != b}
    arrow = (" , then ".join(f"{h['p_wrong'][0] * 100:.1f}% → {h['p_wrong'][1] * 100:.1f}%" for h in hist)
             if hist else "no dot change clearly wins, so the text is kept")
    return _wrap('<div class="h">Before</div>' + _verdict(before["p_wrong"]) + _chips(before)
                 + f'<div class="note" style="text-align:center">↓ {arrow}</div><div class="h">After</div>'
                 + _verdict(after["p_wrong"]) + _pair(scan, after["text"]) + _chips(after, changed))


def compare(img, c1, c2, c3):
    scan = _gray(img)
    cands = [c for c in (c1, c2, c3) if c and c.strip()]
    if not cands:
        raise gr.Error("Type at least one candidate.")
    ranked = CK.pick(scan, cands)
    return _wrap("".join(
        f'<div class="rank{" best" if i == 0 else ""}"><div class="row"><b>{i + 1}</b>'
        f'<span class="txt ar">{html.escape(r["text"])}</span><span class="pill {"bad" if r["p_wrong"] >= 0.5 else "ok"}">'
        f'{r["p_wrong"] * 100:.1f}% wrong</span></div>{_chips(r)}</div>' for i, r in enumerate(ranked)))


def _examples():
    rows = []
    for s in ("test_font_hard", "test_font"):
        man = HERE / "data" / "synth" / s / "manifest.jsonl"
        if man.exists():
            for x in man.read_text(encoding="utf-8").splitlines():
                r = json.loads(x)
                rows.append((str(HERE / "data" / "synth" / s / "scan" / f"{r['id']}.png"), r["cand"], r["true"]))
    return rows


EX = _examples()
# sample text -> (candidate 2 for Compare, answer line); every sample text is unique in the bundled set
TRUTH = {cand: ("" if true == cand else true,
                "**Answer:** this sample text is correct." if true == cand else
                f"**Answer:** this sample text has an error; the true text is <span class='ar' dir='rtl'>{html.escape(true)}</span> "
                "(also put in Compare, candidate 2).") for _, cand, true in EX}

with gr.Blocks(title="Sindhi-Jev checker") as demo:
    gr.Markdown(
        "# Sindhi-Jev checker\n"
        "Does the OCR text match the scan? A 7M-parameter model scores the text and flags letters whose dots look wrong. "
        f"Built for crops of **1–4 printed Sindhi words**. [Docs and results]({REPO}) · images are not stored.")
    with gr.Row():
        with gr.Column(scale=5):
            img = gr.Image(label="Word crop", type="numpy", sources=["upload", "clipboard", "webcam"], height=180)
            with gr.Tabs():
                with gr.Tab("Check"):
                    t1 = gr.Textbox(label="OCR text", rtl=True, lines=1, elem_classes="ar")
                    b1 = gr.Button("Check text", variant="primary")
                with gr.Tab("Repair"):
                    t2 = gr.Textbox(label="OCR text", rtl=True, lines=1)
                    b2 = gr.Button("Repair dots", variant="primary")
                with gr.Tab("Compare"):
                    c1 = gr.Textbox(label="Candidate 1", rtl=True)
                    c2 = gr.Textbox(label="Candidate 2", rtl=True)
                    c3 = gr.Textbox(label="Candidate 3 (optional)", rtl=True)
                    b3 = gr.Button("Compare candidates", variant="primary")
        with gr.Column(scale=6):
            out = gr.HTML('<div class="sj note">Results appear here.</div>')
    if EX:
        # Image + text columns only: in Gradio 6.32 an extra hidden column made the examples table raise a Svelte
        # "each_key_duplicate" error (found 2026-11-26). The true text is looked up from the sample text instead.
        gr.Examples([[p, cand] for p, cand, _ in EX], inputs=[img, t1], examples_per_page=12,
                    label="Synthetic samples in fonts the model never trained on")
        answer = gr.Markdown("")
    # an example fills the Check box; copy it to Repair and Compare, and put the true text in candidate 2
    t1.change(lambda s: (s, s), t1, [t2, c1], show_progress="hidden", queue=False)
    if EX:
        t1.change(lambda s: (TRUTH.get(s, ("", ""))[0], TRUTH.get(s, ("", ""))[1]), t1, [c2, answer],
                  show_progress="hidden", queue=False)
    gr.Markdown(
        "**How to read it.** The percentage is the chance the text does not match the crop; treat it as a ranking score "
        "(calibration error on real words is about 0.11). Darker letter tiles are more suspicious; a red outline means "
        "flagged. On real word scans the model reaches AUROC 0.88–0.90 overall and 0.94 on dot errors; on full lines it "
        "does not work.")
    b1.click(check, [img, t1], out, api_name="check")
    b2.click(repair, [img, t2], out, api_name="repair")
    b3.click(compare, [img, c1, c2, c3], out, api_name="compare")

if __name__ == "__main__":
    demo.launch(css=CSS, head=HEAD, theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"))
