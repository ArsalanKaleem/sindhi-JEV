"""Phase 3 data: vocabulary, image preprocessing, datasets and batching.

Every example (synthetic or real) becomes:
  scan      float32 [1, H, Ws]   the crop, height-normalised, ink = 1, background = 0
  cand_img  float32 [1, H, Wc]   clean render of the candidate in the candidate font
  ids       int64   [L]          candidate code points (vocab ids)
  char_x    float32 [L, 2]       each code point's x-range in cand_img, as fractions of Wc
  seq_label float, char_labels float [L]
Batching pads widths/lengths and returns masks.

Datasets
  OnTheFly   : PairGenerator per index (seeded, reproducible) - training
  Manifest   : fixed synthetic sets written by scripts/build_pairs.py - val / test
  Shards     : pre-generated training pairs (scripts/make_shards.py) when CPU generation is the bottleneck
  RealSet    : real crops + engine candidates (final evaluation only; frozen-set guard in realset.load_real)
"""
import io
import json
import pickle
import random
import unicodedata
from pathlib import Path

import cv2
import numpy as np

from .alphabet import DIACRITICS, EXTRA_CHARS, PUNCT, SINDHI_LETTERS

PAD, UNK, SPACE = 0, 1, 2
_CHARS = sorted(set(SINDHI_LETTERS) | set(EXTRA_CHARS) | set(DIACRITICS) | set(PUNCT)
                | set("0123456789٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹ءٰٔ"))
VOCAB = {c: i + 3 for i, c in enumerate(_CHARS)}
VOCAB_SIZE = len(VOCAB) + 3


def encode(text):
    return [SPACE if c == " " else VOCAB.get(c, UNK) for c in text]


def prep_image(img, height):
    """uint8 grayscale (dark ink on light) -> float32 [1, height, W], ink = 1. Keeps aspect ratio."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    nw = max(8, int(round(w * height / max(h, 1))))
    img = cv2.resize(img, (nw, height), interpolation=cv2.INTER_AREA)
    x = 1.0 - img.astype(np.float32) / 255.0
    return x[None]


def to_example(scan, cand_img, cand, char_x, seq_label, char_labels, cfg, meta=None):
    """Raw arrays -> model example. char_x: [(x0, x1)] in cand_img pixel coordinates (len == len(cand))."""
    H = cfg["img_height"]
    s = prep_image(scan, H)
    c = prep_image(cand_img, H)
    if s.shape[2] > cfg["max_width"]:  # very long crops: squeeze width (rare for 1-4 words)
        s = cv2.resize(s[0], (cfg["max_width"], H), interpolation=cv2.INTER_AREA)[None]
    wc = cand_img.shape[1]
    if c.shape[2] > cfg["max_width"]:
        c = cv2.resize(c[0], (cfg["max_width"], H), interpolation=cv2.INTER_AREA)[None]
    ids = encode(cand)[: cfg["max_chars"]]
    L = len(ids)
    cx = np.array([[a / wc, b / wc] for a, b in char_x[:L]], np.float32).reshape(L, 2)
    return {"scan": s, "cand_img": c, "ids": np.array(ids, np.int64), "char_x": cx,
            "seq_label": float(seq_label), "char_labels": np.array(char_labels[:L], np.float32),
            "meta": meta or {}}


def collate(batch):
    import torch  # noqa: PLC0415
    B = len(batch)
    H = batch[0]["scan"].shape[1]
    ws = max(b["scan"].shape[2] for b in batch)
    wc = max(b["cand_img"].shape[2] for b in batch)
    L = max(max(len(b["ids"]) for b in batch), 1)
    scan = torch.zeros(B, 1, H, ws)
    cand = torch.zeros(B, 1, H, wc)
    scan_w = torch.zeros(B, dtype=torch.long)
    cand_w = torch.zeros(B, dtype=torch.long)
    ids = torch.zeros(B, L, dtype=torch.long)
    char_x = torch.zeros(B, L, 2)
    char_y = torch.zeros(B, L)
    seq_y = torch.zeros(B)
    for i, b in enumerate(batch):
        w = b["scan"].shape[2]
        scan[i, :, :, :w] = torch.from_numpy(b["scan"])
        scan_w[i] = w
        w = b["cand_img"].shape[2]
        cand[i, :, :, :w] = torch.from_numpy(b["cand_img"])
        cand_w[i] = w
        n = len(b["ids"])
        ids[i, :n] = torch.from_numpy(b["ids"])
        char_x[i, :n] = torch.from_numpy(b["char_x"]) * (w / wc)  # fractions of the padded width
        char_y[i, :n] = torch.from_numpy(b["char_labels"])
        seq_y[i] = b["seq_label"]
    return {"scan": scan, "scan_w": scan_w, "cand": cand, "cand_w": cand_w, "ids": ids, "char_mask": ids != PAD,
            "char_x": char_x, "char_y": char_y, "seq_y": seq_y, "meta": [b["meta"] for b in batch]}


def _sample_meta(e):
    groups = sorted({x.get("group", "") for x in e.get("edits", []) if x.get("group")})
    return {"mode": e["mode"], "font": e["font"], "groups": groups, "cand": e["cand"], "true": e["true"]}


class OnTheFly:
    """Infinite-style map dataset: index -> freshly generated pair (seeded by index)."""

    def __init__(self, gen, cfg, seed, split="train", length=10 ** 8):
        self.gen, self.cfg, self.seed, self.split, self.length = gen, cfg, seed, split, length

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        from .pairs import example_rng  # noqa: PLC0415
        e = self.gen.sample(example_rng(self.seed, self.split, i))
        return to_example(e["scan"], e["cand_img"], e["cand"], e["cand_char_x"], e["seq_label"], e["char_labels"],
                          self.cfg, _sample_meta(e))


class Manifest:
    """Fixed synthetic set: data/synth/<set>/{manifest.jsonl, scan/, cand/}."""

    def __init__(self, set_dir, cfg, limit=None):
        self.dir = Path(set_dir)
        with open(self.dir / "manifest.jsonl", encoding="utf-8") as f:
            self.rows = [json.loads(x) for x in f]
        if limit:
            self.rows = self.rows[:limit]
        self.cfg = cfg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        scan = cv2.imread(str(self.dir / "scan" / f"{r['id']}.png"), cv2.IMREAD_GRAYSCALE)
        cand = cv2.imread(str(self.dir / "cand" / f"{r['id']}.png"), cv2.IMREAD_GRAYSCALE)
        meta = _sample_meta(r)
        meta["id"] = r["id"]
        return to_example(scan, cand, r["cand"], r["cand_char_x"], r["seq_label"], r["char_labels"], self.cfg, meta)


def write_shard(path, examples):
    """examples: generator dicts from PairGenerator.sample -> one pickle of PNG-compressed pairs."""
    out = []
    for e in examples:
        ok1, s = cv2.imencode(".png", e["scan"])
        ok2, c = cv2.imencode(".png", e["cand_img"])
        assert ok1 and ok2
        out.append({"scan": s.tobytes(), "cand_img": c.tobytes(), "cand": e["cand"], "char_x": e["cand_char_x"],
                    "seq_label": e["seq_label"], "char_labels": e["char_labels"], "meta": _sample_meta(e)})
    with open(path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)


class Shards:
    """Pre-generated training pairs (list of shard pickles), all loaded into RAM (PNG bytes are small)."""

    def __init__(self, paths, cfg):
        self.items = []
        for p in paths:
            with open(p, "rb") as f:
                self.items += pickle.load(f)
        self.cfg = cfg

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        r = self.items[i]
        dec = lambda b: cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE)  # noqa: E731
        return to_example(dec(r["scan"]), dec(r["cand_img"]), r["cand"], r["char_x"], r["seq_label"],
                          r["char_labels"], self.cfg, r["meta"])


def clean_candidate(s):
    """Same normalisation as the real-set labels (realset.norm) so char positions line up."""
    from .realset import norm  # noqa: PLC0415
    return norm(unicodedata.normalize("NFC", s or ""))


class RealSet:
    """Real crops x engine candidates with derived labels (rows of candidate_labels.csv)."""

    def __init__(self, root, label_rows, cand_rows, crop_rows, cfg, renderer):
        self.root = Path(root)
        crops = {r["crop_id"]: r for r in crop_rows}
        cands = {(r["crop_id"], r["engine"]): r for r in cand_rows}
        self.rows = []
        for lr in label_rows:
            c = cands[(lr["crop_id"], lr["engine"])]
            self.rows.append({**lr, "cand_text": clean_candidate(c["cand_text"]), "seq_conf": c["seq_conf"],
                              "char_conf_json": c["char_conf_json"], "image_path": crops[lr["crop_id"]]["image_path"],
                              "stratum": crops[lr["crop_id"]]["stratum"], "source_id": crops[lr["crop_id"]]["source_id"]})
        self.cfg, self.r = cfg, renderer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        scan = cv2.imread(str(self.root / r["image_path"]), cv2.IMREAD_GRAYSCALE)
        text = r["cand_text"] or " "
        cand_img, char_x = self.r.render(text)
        labels = json.loads(r["char_labels_json"])
        if len(labels) != len(text):  # empty candidate -> one dummy position, sequence label still counts
            labels = (labels + [0] * len(text))[: len(text)]
        meta = {k: r[k] for k in ("crop_id", "engine", "stratum", "source_id")}
        meta["errors"] = json.loads(r.get("errors_json") or "[]")
        meta["ambiguous"] = json.loads(r.get("ambiguous_cand_json") or "[]")
        return to_example(scan, cand_img, text, char_x, int(r["seq_label"]), labels, self.cfg, meta)


def seed_worker(worker_id):
    import torch  # noqa: PLC0415
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)
