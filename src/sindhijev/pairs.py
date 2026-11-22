"""Synthetic (scan, candidate, candidate-render, labels) pairs.

One example:
  true text  -> rendered in a SCAN font at a random size -> degraded = "scan"
  candidate  -> true text or a corrupted copy (corrupt.make_candidate)
  candidate  -> rendered cleanly in the fixed CANDIDATE font (what a deployed
                verifier would do: it never knows the scan font)
  labels     -> seq_label = 1 if the candidate is wrong; char_labels per
                candidate code point

Guard: if the corrupted candidate renders pixel-identically to the true
text in the scan font, the error is invisible, so the example is relabelled
positive (counted in `relabelled`).

The same class serves fixed val/test sets (scripts/build_pairs.py) and
on-the-fly training data on Kaggle (seeded per index, so reproducible).
"""
import json
import random
from pathlib import Path

import cv2
import numpy as np

from .corrupt import load_groups, make_candidate
from .degrade import degrade
from .render import Renderer

ROOT = Path(__file__).resolve().parents[2]


def load_spans(path, split):
    with open(path, encoding="utf-8") as f:
        return [r["text"] for r in map(json.loads, f) if r["split"] == split]


class PairGenerator:
    def __init__(self, cfg, split, scan_fonts, root=ROOT, spans=None):
        self.cfg = cfg
        self.root = Path(root)
        self.spans = spans if spans is not None else load_spans(self.root / cfg["spans"], split)
        self.groups = load_groups(self.root / cfg["confusion_set"])
        self.scan_fonts = list(scan_fonts)
        self._scan = {}
        self.cand_r = Renderer(self.root / "fonts" / cfg["candidate_font"], size=cfg["candidate_size"])
        self.relabelled = 0
        self.lexicon = set()
        if cfg.get("lexicon"):  # train-split word list (scripts/build_lexicon.py); used by mode 'realword'
            self.lexicon = set(json.loads((self.root / cfg["lexicon"]).read_text(encoding="utf-8")))

    def _scan_renderer(self, font, size):
        key = (font, size)
        if key not in self._scan:
            self._scan[key] = Renderer(self.root / "fonts" / font, size=size)
        return self._scan[key]

    def text_window(self, rng):
        words = rng.choice(self.spans).split()
        k = rng.randint(1, min(self.cfg["max_words"], len(words)))
        s = rng.randint(0, len(words) - k)
        return " ".join(words[s:s + k])

    def sample(self, rng, mode=None):
        if not self.lexicon:  # data_v1 path, unchanged (the fixed v1 sets are reproducible from it)
            text = self.text_window(rng)
            font = rng.choice(self.scan_fonts)
            size = rng.randint(*self.cfg["scan_size"])
            sr = self._scan_renderer(font, size)
            c = make_candidate(text, self.groups, rng, mode=mode, mix=self.cfg["mix"])
            return self._finish(rng, text, font, size, sr, c)
        if mode is None:
            mode = rng.choices(list(self.cfg["mix"]), weights=list(self.cfg["mix"].values()))[0]
        for _ in range(50):  # real-word swaps exist only in some texts: resample the text, keep the mode
            text = self.text_window(rng)
            c = make_candidate(text, self.groups, rng, mode=mode, mix=self.cfg["mix"], lexicon=self.lexicon)
            if c["mode"] != "none":
                break
        else:
            c = make_candidate(text, self.groups, rng, mode="confusion", mix=self.cfg["mix"])
        font = rng.choice(self.scan_fonts)
        size = rng.randint(*self.cfg["scan_size"])
        sr = self._scan_renderer(font, size)
        if self.cfg.get("realism"):  # v3: some fonts lack a few code points -> pick one that covers both texts
            for _ in range(20):
                if sr.covers(text) and sr.covers(c["cand"]):
                    break
                font = rng.choice(self.scan_fonts)
                sr = self._scan_renderer(font, size)
        return self._finish(rng, text, font, size, sr, c)

    def _context_text(self, rng, sr):
        for _ in range(10):
            t = self.text_window(rng)
            if sr.covers(t):
                return t
        return None

    def _realism(self, rng, clean, sr, size):
        """data_v3 only. Changes the page around the text, never the text's own dots:
        horizontal stretch, slivers of the neighbouring lines at the top/bottom edge (as in real crops),
        faint mirrored bleed-through from the other side of the page."""
        p = {}
        h, w = clean.shape
        if rng.random() < 0.4:
            f = rng.uniform(0.8, 1.25)
            clean = cv2.resize(clean, (max(8, int(w * f)), h), interpolation=cv2.INTER_AREA)
            h, w = clean.shape
            p["stretch"] = round(f, 2)
        if rng.random() < 0.5:
            parts = [clean]
            for side in ("top", "bottom"):
                if rng.random() < 0.6:
                    t = self._context_text(rng, sr)
                    if t is None:
                        continue
                    other, _ = sr.render(t, pad=0)
                    keep = max(2, int(other.shape[0] * rng.uniform(0.1, 0.35)))
                    sl = other[-keep:] if side == "top" else other[:keep]
                    canvas = np.full((keep, w), 255, np.uint8)
                    x0 = rng.randint(-sl.shape[1] // 2, max(0, w - sl.shape[1] // 2))
                    a, b = max(0, x0), min(w, x0 + sl.shape[1])
                    if b > a:
                        canvas[:, a:b] = sl[:, a - x0:b - x0]
                    gap = np.full((rng.randint(0, max(1, size // 6)), w), 255, np.uint8)
                    parts = [canvas, gap] + parts if side == "top" else parts + [gap, canvas]
                    p[f"sliver_{side}"] = 1
            clean = np.vstack(parts)
            h, w = clean.shape
        if rng.random() < 0.2:
            t = self._context_text(rng, sr)
            if t is not None:
                back, _ = sr.render(t, pad=0)
                back = cv2.resize(back[:, ::-1], (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
                alpha = rng.uniform(0.08, 0.25)
                clean = np.minimum(clean.astype(np.float32), 255 - alpha * (255 - back)).astype(np.uint8)
                p["bleed"] = round(alpha, 2)
        return clean, p

    def _finish(self, rng, text, font, size, sr, c):
        clean, _ = sr.render(text, pad=rng.randint(2, size // 3), ink=rng.randint(0, 50))
        real_p = {}
        if self.cfg.get("realism"):
            clean, real_p = self._realism(rng, clean, sr, size)
        if c["mode"] != "positive":
            cand_in_scan_font, _ = sr.render(c["cand"], pad=0)
            true_in_scan_font, _ = sr.render(text, pad=0)
            if cand_in_scan_font.shape == true_in_scan_font.shape and \
                    np.array_equal(cand_in_scan_font, true_in_scan_font):
                self.relabelled += 1
                c = {"cand": c["cand"], "labels": [0] * len(c["cand"]), "mode": "invisible", "edits": c["edits"]}
        scan, dparams = degrade(clean, rng, max_severity=self.cfg["max_severity"],
                                realism=bool(self.cfg.get("realism")))
        dparams.update(real_p)
        cand_img, char_x = self.cand_r.render(c["cand"])
        return {
            "true": text, "cand": c["cand"], "mode": c["mode"], "edits": c["edits"],
            "seq_label": int(any(c["labels"])), "char_labels": c["labels"],
            "font": font, "size": size, "degrade": dparams,
            "scan": scan, "cand_img": cand_img, "cand_char_x": char_x,
        }


def example_rng(seed, split, index):
    """Independent, reproducible RNG per (split, index), safe across DataLoader workers."""
    return random.Random(f"{seed}:{split}:{index}")
