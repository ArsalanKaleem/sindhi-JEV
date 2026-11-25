"""CPU tool: check an OCR candidate against its crop, or pick the best of several OCR outputs (ONNX, no torch).

    from sindhijev.tool import Checker
    ck = Checker()                                  # downloads the ONNX weights on first use (sha256-checked)
    r = ck.check("crop.png", "سنڌ جي تاريخ")        # {"p_wrong": ..., "suspicious": [...], "chars": [...]}
    best = ck.pick("crop.png", ["candidate 1", "candidate 2"])

p_wrong is the model's probability that the text does NOT match the image (temperature-scaled on synthetic
validation data). A letter is flagged when its score is above the char threshold chosen on synthetic validation.
Both numbers come from the model card (MODEL.json next to the weights), never from the real test set.
"""
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unicodedata
import urllib.request
from importlib import resources
from pathlib import Path

import cv2
import numpy as np

from .data import encode, fill_ranges, prep_image
from .realset import norm

# Filled in when a GitHub release with the weights exists (see docs/release.md); override with SINDHIJEV_WEIGHTS.
WEIGHTS_URL = os.environ.get("SINDHIJEV_WEIGHTS_URL", "")
WEIGHTS_SHA256 = os.environ.get("SINDHIJEV_WEIGHTS_SHA256", "")
CACHE = Path(os.environ.get("SINDHIJEV_CACHE", Path.home() / ".cache" / "sindhijev"))
TESS_ENGINES = {"tess_snd": "snd", "tess_urd": "urd", "tess_ara": "ara", "tess_fas": "fas"}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weights_path():
    """Local ONNX file: SINDHIJEV_WEIGHTS, else the cached download (fetched once, checksum verified)."""
    if os.environ.get("SINDHIJEV_WEIGHTS"):
        return Path(os.environ["SINDHIJEV_WEIGHTS"])
    dst = CACHE / "sindhijev_v2.onnx"
    if dst.exists():
        return dst
    if not WEIGHTS_URL:
        raise FileNotFoundError("No weights: set SINDHIJEV_WEIGHTS=/path/to/sindhijev_v2.onnx "
                                "(the download URL is set once the weights are released).")
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    with urllib.request.urlopen(WEIGHTS_URL, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    if WEIGHTS_SHA256 and _sha256(tmp) != WEIGHTS_SHA256:
        tmp.unlink()
        raise RuntimeError("downloaded weights fail the sha256 check")
    tmp.replace(dst)
    for side in ("json",):  # model card next to the weights
        try:
            urllib.request.urlretrieve(WEIGHTS_URL.rsplit(".", 1)[0] + "." + side, dst.with_suffix("." + side))
        except OSError:
            pass
    return dst


def _font():
    return resources.files("sindhijev") / "assets" / "NotoNaskhArabic.ttf"


_PARTNERS = None


def _partners():
    """letter -> dot-variant letters it was reviewed as confusable with (single code points, no encoding groups)."""
    global _PARTNERS
    if _PARTNERS is None:
        cs = json.loads((resources.files("sindhijev") / "assets" / "confusion_set.json").read_text(encoding="utf-8"))
        _PARTNERS = {}
        for g in cs["groups"]:
            if g["kind"] == "encoding":
                continue
            for p in g.get("pairs", []):
                if len(p["a"]) == 1 and len(p["b"]) == 1:
                    _PARTNERS.setdefault(p["a"], []).append(p["b"])
                    _PARTNERS.setdefault(p["b"], []).append(p["a"])
    return _PARTNERS


class Checker:
    def __init__(self, weights=None, threads=None):
        import onnxruntime as ort  # noqa: PLC0415
        from .render import Renderer  # noqa: PLC0415
        w = Path(weights) if weights else weights_path()
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(w), so, providers=["CPUExecutionProvider"])
        card = w.with_suffix(".json")
        meta = json.loads(card.read_text(encoding="utf-8")) if card.exists() else {}
        cfg = meta.get("model_cfg", {})
        self.H = cfg.get("img_height", 48)
        self.max_w = cfg.get("max_width", 1024)
        self.max_chars = cfg.get("max_chars", 96)
        self.temperature = float(meta.get("temperature", 1.0))
        self.char_thr = float(meta.get("char_threshold", 0.5))
        with resources.as_file(_font()) as fp:
            self.r = Renderer(fp, size=cfg.get("candidate_size", 48))

    @staticmethod
    def _load(image):
        if isinstance(image, np.ndarray):
            img = image
        else:
            img = cv2.imdecode(np.fromfile(str(image), np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"cannot read image {image}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _example(self, scan, text):
        text = norm(unicodedata.normalize("NFC", text or "")) or " "
        cand_img, char_x = self.r.render(text)
        s, c = prep_image(scan, self.H), prep_image(cand_img, self.H)
        if s.shape[2] > self.max_w:
            s = cv2.resize(s[0], (self.max_w, self.H), interpolation=cv2.INTER_AREA)[None]
        if c.shape[2] > self.max_w:
            c = cv2.resize(c[0], (self.max_w, self.H), interpolation=cv2.INTER_AREA)[None]
        ids = encode(text)[: self.max_chars]
        wc = cand_img.shape[1]
        cx = np.array([[a / wc, b / wc] for a, b in fill_ranges(char_x[: len(ids)])], np.float32).reshape(-1, 2)
        return text, s, c, np.array(ids, np.int64), cx

    def _run(self, exs):
        B, H = len(exs), self.H
        ws, wc = max(e[1].shape[2] for e in exs), max(e[2].shape[2] for e in exs)
        L = max(max(len(e[3]) for e in exs), 1)
        f = {"scan": np.zeros((B, 1, H, ws), np.float32), "scan_w": np.zeros(B, np.int64),
             "cand": np.zeros((B, 1, H, wc), np.float32), "cand_w": np.zeros(B, np.int64),
             "ids": np.zeros((B, L), np.int64), "char_x": np.zeros((B, L, 2), np.float32)}
        for i, (_, s, c, ids, cx) in enumerate(exs):
            f["scan"][i, :, :, : s.shape[2]] = s
            f["scan_w"][i] = s.shape[2]
            f["cand"][i, :, :, : c.shape[2]] = c
            f["cand_w"][i] = c.shape[2]
            f["ids"][i, : len(ids)] = ids
            f["char_x"][i, : len(ids)] = cx * (c.shape[2] / wc)  # fractions of the padded width (as in training)
        seq, ch = self.sess.run(None, f)
        return seq, ch

    def check_many(self, image, texts):
        scan = self._load(image)
        exs = [self._example(scan, t) for t in texts]
        # one forward pass per candidate: in a padded batch, BatchNorm turns the padding into non-zero activations
        # that reach the last real tokens (the start of a right-to-left word), so a score would depend on the other
        # items in the batch (median 0.02, up to 1.3 logit on synthetic data; found 2026-11-26).
        runs = [self._run([e]) for e in exs]
        out = []
        for i, (text, _, _, ids, _) in enumerate(exs):
            seq_i, ch_i = runs[i]
            p = float(_sigmoid(seq_i[0] / self.temperature))
            cs = _sigmoid(ch_i[0, : len(ids)]).tolist()
            out.append({"text": text, "p_wrong": p,
                        "chars": [{"i": k, "char": text[k], "score": round(v, 4)} for k, v in enumerate(cs)],
                        "suspicious": [k for k, v in enumerate(cs) if v > self.char_thr and text[k] != " "]})
        return out

    def check(self, image, text):
        return self.check_many(image, [text])[0]

    def repair(self, image, text, margin=0.6, top_k=3, max_edits=2):
        """Dot repair: for the top_k most suspicious letters, try their reviewed confusion partners (dot variants of
        the same skeleton), re-score, and keep a swap only if p_wrong drops by at least `margin` (chosen on synthetic
        val, results/dot_repair.json). Repeats up to max_edits times. Returns (text, history)."""
        partners = _partners()
        scan = self._load(image)
        cur = self.check_many(scan, [text])[0]
        hist = []
        for _ in range(max_edits):
            pos = sorted(range(len(cur["text"])), key=lambda k: -cur["chars"][k]["score"])[:top_k]
            variants = [cur["text"][:k] + b + cur["text"][k + 1:] for k in pos
                        if cur["chars"][k]["score"] > self.char_thr for b in partners.get(cur["text"][k], [])]
            if not variants:
                break
            best = min(self.check_many(scan, variants), key=lambda r: r["p_wrong"])
            if best["p_wrong"] > cur["p_wrong"] - margin:
                break
            hist.append({"from": cur["text"], "to": best["text"], "p_wrong": [cur["p_wrong"], best["p_wrong"]]})
            cur = best
        return cur["text"], hist

    def pick(self, image, candidates):
        """Score every candidate on the same crop and return them best first (lowest p_wrong)."""
        cands = [c for c in dict.fromkeys(candidates) if c is not None]
        if not cands:
            raise ValueError("no candidates")
        return sorted(self.check_many(image, cands), key=lambda r: r["p_wrong"])


def tesseract_candidates(image, engines=None, tess="tesseract", psm=7):
    """Run the installed Tesseract language models on the crop (single line). Missing models are skipped."""
    if not shutil.which(tess):
        raise FileNotFoundError("tesseract not found on PATH (or pass --tesseract)")
    langs = set(subprocess.run([tess, "--list-langs"], capture_output=True, text=True).stdout.split())
    out = {}
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "crop.png"
        cv2.imwrite(str(p), Checker._load(image))
        for name, lang in (engines or TESS_ENGINES).items():
            if lang not in langs:
                continue
            r = subprocess.run([tess, str(p), "stdout", "-l", lang, "--psm", str(psm)],
                               capture_output=True, text=True, encoding="utf-8")
            t = " ".join(r.stdout.split())
            if t:
                out[name] = t
    return out
