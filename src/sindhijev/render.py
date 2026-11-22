"""HarfBuzz-shaped, FreeType-rasterised rendering of Sindhi text.

Works the same on Windows and Kaggle (no libraqm needed). Output is a
grayscale uint8 array, black text on white, plus a per-character x-range
(logical index -> (x0, x1) in pixels) taken from HarfBuzz clusters, so the
candidate image can be tied to candidate characters.

Input text must be a single RTL run (Arabic script, spaces, Arabic
punctuation); Latin/digit runs are not bidi-reordered.
"""
import freetype
import numpy as np
import uharfbuzz as hb


class Renderer:
    def __init__(self, font_path, size=48, language="snd"):
        self.path = str(font_path)
        self.size = size
        self.language = language
        blob = hb.Blob.from_file_path(self.path)
        self.hb_face = hb.Face(blob)
        self.hb_font = hb.Font(self.hb_face)
        self.upem = self.hb_face.upem
        self.ft = freetype.Face(self.path)
        self.ft.set_pixel_sizes(0, size)
        self.scale = size / self.upem
        self.cmap = self.hb_face.unicodes

    def covers(self, text):
        return all(ord(c) in self.cmap or c in " ‌‍" for c in text)

    def shape(self, text):
        buf = hb.Buffer()
        buf.add_codepoints([ord(c) for c in text])
        buf.direction = "rtl"
        buf.script = "Arab"
        buf.language = self.language
        hb.shape(self.hb_font, buf, {})
        return buf.glyph_infos, buf.glyph_positions

    def render(self, text, pad=None, ink=0):
        """Return (img uint8 HxW, char_x: list of (x0, x1) or None per char)."""
        infos, poss = self.shape(text)
        pad = self.size // 4 if pad is None else pad
        glyphs, pen_x = [], 0.0
        for info, pos in zip(infos, poss):  # visual order, left to right
            self.ft.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_NO_HINTING)
            g = self.ft.glyph
            bm = g.bitmap
            arr = np.array(bm.buffer, dtype=np.uint8).reshape(bm.rows, bm.pitch)[:, :bm.width] \
                if bm.rows and bm.width else np.zeros((0, 0), np.uint8)
            x = pen_x + pos.x_offset * self.scale + g.bitmap_left
            y = -(pos.y_offset * self.scale) - g.bitmap_top  # baseline at y=0, y grows down
            adv = pos.x_advance * self.scale
            glyphs.append((arr, x, y, info.cluster, pen_x, pen_x + adv))
            pen_x += adv
        inked = [(a, x, y) for a, x, y, *_ in glyphs if a.size]
        if not inked:
            return np.full((self.size + 2 * pad, max(1, int(pen_x)) + 2 * pad), 255, np.uint8), [None] * len(text)
        x0 = min(min(x for _, x, _ in inked), 0.0)
        x1 = max(max(x + a.shape[1] for a, x, _ in inked), pen_x)
        y0 = min(y for _, _, y in inked)
        y1 = max(y + a.shape[0] for a, _, y in inked)
        W, H = int(np.ceil(x1 - x0)) + 2 * pad, int(np.ceil(y1 - y0)) + 2 * pad
        cov = np.zeros((H, W), np.float32)
        char_x = [None] * len(text)
        for arr, x, y, cl, ax0, ax1 in glyphs:
            if arr.size:
                xi, yi = int(round(x - x0)) + pad, int(round(y - y0)) + pad
                h, w = arr.shape
                region = cov[yi:yi + h, xi:xi + w]
                np.maximum(region, arr[:region.shape[0], :region.shape[1]] / 255.0, out=region)
                lo, hi = xi, xi + w
            else:
                lo, hi = int(ax0 - x0) + pad, int(ax1 - x0) + pad
            if cl < len(text):
                prev = char_x[cl]
                char_x[cl] = (lo, hi) if prev is None else (min(prev[0], lo), max(prev[1], hi))
        # Characters merged into a ligature cluster inherit the cluster's range.
        last = None
        for i in range(len(text)):
            if char_x[i] is None:
                char_x[i] = last
            else:
                last = char_x[i]
        img = (255 - cov * (255 - ink)).clip(0, 255).astype(np.uint8)
        return img, char_x
