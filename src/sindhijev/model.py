"""Sindhi-Jev verifier.

    scan crop ──CNN──> scan tokens  ─┐
    candidate render ──CNN──> render tokens ─┤ (memory, + type embeddings + 1-D x position)
                                             │
    [CLS] + one query per candidate code point ──> transformer decoder (self-attn over queries,
        query_i = char embedding + index position          cross-attn to memory)
                + render features pooled over char i's x-range ("char-aligned render")
                                             │
                     CLS -> sequence logit (candidate wrong?)
                 query_i -> char logit (code point i has a dot/char error?)

Ablation switches (config): use_scan, use_render, char_align, char_head.
  use_scan=False  -> a text+render-only model (cannot see the crop): sanity floor
  use_render=False -> scan + text only (no re-render)
  char_align=False -> queries do not get pooled render features
Size with the default config is ~12M parameters (budget: <= 60M).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import PAD, VOCAB_SIZE

DEFAULT = {
    "img_height": 48, "max_width": 1024, "max_chars": 96,
    "d_model": 256, "n_heads": 8, "n_layers": 4, "ffn": 1024, "dropout": 0.1,
    "cnn_channels": [32, 64, 128, 256],
    "use_scan": True, "use_render": True, "char_align": True, "char_head": True,
    "char_loss_weight": 1.0, "seq_pos_weight": 1.0, "char_pos_weight": 5.0,
}


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, pool):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = nn.Conv2d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()
        self.pool = pool

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        y = F.relu(y + self.skip(x))
        return F.max_pool2d(y, self.pool) if self.pool else y


class Encoder(nn.Module):
    """[B,1,48,W] -> tokens [B, W/4, d] (height pooled away, width stride 4: ~3 tokens per letter at 48 px)."""

    def __init__(self, ch, d):
        super().__init__()
        pools = [(2, 2), (2, 2), (2, 1), (2, 1)]
        layers, cin = [], 1
        for c, p in zip(ch, pools):
            layers.append(ConvBlock(cin, c, p))
            cin = c
        self.net = nn.Sequential(*layers)
        self.proj = nn.Linear(cin * 3, d)  # 48 / 16 = 3 rows kept as features (above / on / below baseline)

    def forward(self, x):
        f = self.net(x)  # [B, C, 3, W/4]
        B, C, h, w = f.shape
        return self.proj(f.permute(0, 3, 1, 2).reshape(B, w, C * h))


def sinusoid(n, d, device):
    pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(0, d, 2, device=device, dtype=torch.float32)
    ang = pos / (10000 ** (i / d))
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(ang), torch.cos(ang)
    return pe


class Verifier(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = {**DEFAULT, **(cfg or {})}
        c, d = self.cfg, self.cfg["d_model"]
        self.scan_enc = Encoder(c["cnn_channels"], d) if c["use_scan"] else None
        self.rend_enc = Encoder(c["cnn_channels"], d) if (c["use_render"] or c["char_align"]) else None
        self.type_emb = nn.Embedding(2, d)  # 0 = scan token, 1 = render token
        self.char_emb = nn.Embedding(VOCAB_SIZE, d, padding_idx=PAD)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.align_proj = nn.Linear(d, d) if c["char_align"] else None
        layer = nn.TransformerDecoderLayer(d, c["n_heads"], c["ffn"], c["dropout"], batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, c["n_layers"])
        self.norm = nn.LayerNorm(d)
        self.seq_head = nn.Linear(d, 1)
        self.char_head = nn.Linear(d, 1) if c["char_head"] else None
        if not c["use_scan"] and not c["use_render"]:  # text-only: memory is a single learned token
            self.null_mem = nn.Parameter(torch.zeros(1, 1, d))

    @staticmethod
    def _mask_from_width(tokens, widths, stride=4):
        n = tokens.shape[1]
        lens = torch.div(widths + stride - 1, stride, rounding_mode="floor").clamp(max=n)
        return torch.arange(n, device=tokens.device)[None] >= lens[:, None]  # True = padding

    def _pool_render(self, rtok, char_x, rmask):
        """Mean of render tokens whose x-centre falls in each char's [x0, x1] (fractions of padded width)."""
        B, n, d = rtok.shape
        centres = (torch.arange(n, device=rtok.device, dtype=torch.float32) + 0.5) / n  # [n]
        x0, x1 = char_x[..., 0:1], char_x[..., 1:2]  # [B, L, 1]
        # widen tiny ranges (diacritics, narrow letters) to at least one token
        half = 0.5 / n
        lo, hi = torch.minimum(x0, (x0 + x1) / 2 - half), torch.maximum(x1, (x0 + x1) / 2 + half)
        w = ((centres[None, None] >= lo) & (centres[None, None] <= hi) & ~rmask[:, None]).float()  # [B, L, n]
        w = w / w.sum(-1, keepdim=True).clamp(min=1.0)
        return torch.bmm(w, rtok)

    def forward(self, b):
        c = self.cfg
        dev = b["ids"].device
        mem, mmask = [], []
        rtok = rmask = None
        if self.scan_enc is not None:
            s = self.scan_enc(b["scan"])
            s = s + sinusoid(s.shape[1], s.shape[2], dev)[None] + self.type_emb.weight[0]
            mem.append(s)
            mmask.append(self._mask_from_width(s, b["scan_w"]))
        if self.rend_enc is not None:
            rtok = self.rend_enc(b["cand"])
            rmask = self._mask_from_width(rtok, b["cand_w"])
            if c["use_render"]:
                mem.append(rtok + sinusoid(rtok.shape[1], rtok.shape[2], dev)[None] + self.type_emb.weight[1])
                mmask.append(rmask)
        if not mem:
            B = b["ids"].shape[0]
            mem, mmask = [self.null_mem.expand(B, -1, -1)], [torch.zeros(B, 1, dtype=torch.bool, device=dev)]
        memory, memory_mask = torch.cat(mem, 1), torch.cat(mmask, 1)

        ids = b["ids"]
        B, L = ids.shape
        q = self.char_emb(ids) + sinusoid(L, c["d_model"], dev)[None]
        if self.align_proj is not None:
            q = q + self.align_proj(self._pool_render(rtok, b["char_x"], rmask))
        q = torch.cat([self.cls.expand(B, -1, -1), q], 1)
        qmask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=dev), ids == PAD], 1)
        h = self.norm(self.dec(q, memory, tgt_key_padding_mask=qmask, memory_key_padding_mask=memory_mask))
        out = {"seq_logit": self.seq_head(h[:, 0]).squeeze(-1)}
        if self.char_head is not None:
            out["char_logit"] = self.char_head(h[:, 1:]).squeeze(-1)
        return out

    def loss(self, out, b):
        c = self.cfg
        dev = out["seq_logit"].device
        seq = F.binary_cross_entropy_with_logits(out["seq_logit"], b["seq_y"],
                                                 pos_weight=torch.tensor(c["seq_pos_weight"], device=dev))
        total, parts = seq, {"seq": seq.detach()}
        if "char_logit" in out and c["char_loss_weight"] > 0:
            m = b["char_mask"].float()
            per = F.binary_cross_entropy_with_logits(out["char_logit"], b["char_y"], reduction="none",
                                                     pos_weight=torch.tensor(c["char_pos_weight"], device=dev))
            ch = (per * m).sum() / m.sum().clamp(min=1)
            total = total + c["char_loss_weight"] * ch
            parts["char"] = ch.detach()
        return total, parts


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def to_device(b, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def warmup_cosine(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1 + math.cos(math.pi * p))
