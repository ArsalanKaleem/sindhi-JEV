"""Cheap baselines (CPU only, no GPU hours).

(a) OCR confidence   : seq score = 1 - engine confidence; char score = 1 - per-char confidence (real set only)
(b) text-only checker: character 5-gram LM (forward + backward, Witten-Bell interpolation) + word dictionary,
                       both trained on the TRAIN text split only. Features -> logistic regression fitted on the
                       synthetic VAL set (never on the real set).
(c) multi-engine voting: a candidate is suspicious if the other engines disagree with it; per char, the share
                       of other engines whose alignment does not confirm that character (real set only)
Also: is_nonword_error() to split results into non-word vs real-word errors.
"""
import json
import math
import pickle
from collections import Counter, defaultdict

import numpy as np

from .align import levenshtein_ops

BOS, EOS = "\x02", "\x03"


class CharLM:
    """Order-n character LM with Witten-Bell interpolation down to a uniform distribution."""

    def __init__(self, order=5):
        self.n = order
        self.counts = [defaultdict(Counter) for _ in range(order)]  # k -> context(len k) -> next-char counts
        self.vocab = set()

    def fit(self, texts):
        for t in texts:
            s = BOS * (self.n - 1) + t + EOS
            self.vocab.update(t + EOS)
            for i in range(self.n - 1, len(s)):
                for k in range(self.n):
                    self.counts[k][s[i - k:i]][s[i]] += 1
        self.V = len(self.vocab) + 1
        # freeze to plain dicts with totals and distinct-type counts for speed / pickling
        self.tab = [{ctx: (dict(c), sum(c.values()), len(c)) for ctx, c in lvl.items()} for lvl in self.counts]
        del self.counts
        return self

    def prob(self, ctx, ch):
        p = 1.0 / self.V
        for k in range(0, self.n):
            c = ctx[len(ctx) - k:] if k else ""
            if len(c) < k:
                break
            e = self.tab[k].get(c)
            if e is None:
                continue
            cnt, tot, types = e
            lam = tot / (tot + types)
            p = lam * cnt.get(ch, 0) / tot + (1 - lam) * p
        return p

    def surprisal(self, text):
        """-log2 P(c_i | previous n-1 chars) for each char of text."""
        s = BOS * (self.n - 1) + text
        return [-math.log2(self.prob(s[i - self.n + 1:i], s[i])) for i in range(self.n - 1, len(s))]


def _logistic_fit(X, y, l2=1e-3, iters=500, lr=0.5):
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0]
        w -= lr * g
    return {"mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist()}


def _logistic_predict(m, X):
    Z = np.c_[(np.asarray(X, np.float64) - m["mu"]) / m["sd"], np.ones(len(X))]
    return (1 / (1 + np.exp(-Z @ np.asarray(m["w"])))).tolist()


class TextChecker:
    """Baseline (b). Sees only the candidate string, never the image."""

    def __init__(self, order=5, min_count=2):
        self.order, self.min_count = order, min_count

    def fit_lm(self, texts):
        texts = list(texts)
        self.fwd = CharLM(self.order).fit(texts)
        self.bwd = CharLM(self.order).fit([t[::-1] for t in texts])
        wc = Counter(w for t in texts for w in t.split())
        self.dict = {w for w, c in wc.items() if c >= self.min_count}
        return self

    def char_features(self, cand):
        if not cand:
            return []
        f = self.fwd.surprisal(cand)
        b = self.bwd.surprisal(cand[::-1])[::-1]
        oov = []
        for w in cand.split(" "):
            oov += [float(w not in self.dict)] * len(w) + [0.0]
        oov = oov[:len(cand)]
        return [[fi, bi, fi + bi, o] for fi, bi, o in zip(f, b, oov)]

    def seq_features(self, cand):
        cf = self.char_features(cand)
        if not cf:
            return [0.0, 0.0, 0.0, 1.0, 0.0]
        tot = [c[2] for c in cf]
        words = cand.split()
        oov = sum(w not in self.dict for w in words) / max(1, len(words))
        return [max(tot), sum(tot) / len(tot), oov, float(oov > 0), math.log(1 + len(cand))]

    def fit_heads(self, recs):
        """recs: [{'cand', 'seq_y', 'char_y'}] from the SYNTHETIC val set."""
        self.seq_m = _logistic_fit([self.seq_features(r["cand"]) for r in recs], [r["seq_y"] for r in recs])
        Xc, yc = [], []
        for r in recs:
            for f, y in zip(self.char_features(r["cand"]), r["char_y"]):
                Xc.append(f)
                yc.append(y)
        self.char_m = _logistic_fit(Xc, yc)
        return self

    def score(self, cand):
        seq = _logistic_predict(self.seq_m, [self.seq_features(cand)])[0]
        cf = self.char_features(cand)
        ch = _logistic_predict(self.char_m, cf) if cf else []
        return seq, ch

    def is_word(self, w):
        return w in self.dict

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


def ocr_confidence(cand, seq_conf, char_conf_json):
    """Baseline (a). Missing confidence -> 0.5 (uninformative)."""
    seq = 1 - float(seq_conf) if seq_conf not in ("", None) else 0.5
    try:
        d = json.loads(char_conf_json) if char_conf_json else {}
    except json.JSONDecodeError:
        d = {}
    chars = d.get("chars") or []
    if len(chars) == len(cand) and chars:
        ch = [1 - c if c is not None else 0.0 for c in chars]
    else:  # fall back to word confidences broadcast over each word, else the sequence score
        words = d.get("words") or []
        toks = cand.split(" ")
        if len(words) == len(toks):
            ch = []
            for w, c in zip(toks, words):
                ch += [1 - c if c is not None else seq] * len(w) + [0.0]
            ch = ch[:len(cand)]
        else:
            ch = [seq] * len(cand)
    return seq, ch


def voting(cand, others):
    """Baseline (c). others: the other engines' outputs for the same crop (normalised)."""
    others = [o for o in others if o is not None]
    if not others:
        return 0.5, [0.5] * len(cand)
    seq = sum(o != cand for o in others) / len(others)
    ch = [0.0] * len(cand)
    for o in others:
        confirmed = [False] * len(cand)
        for op, gi, cj in levenshtein_ops(o, cand):
            if op == "eq":
                confirmed[cj] = True
        for k in range(len(cand)):
            ch[k] += (not confirmed[k]) / len(others)
    return seq, ch


def is_nonword_error(cand, gold, checker):
    """True if a wrong candidate contains a word that is not in the dictionary while the gold word is
    (a spell-checker could catch it); False = real-word error (every wrong word is itself a valid word)."""
    cw, gw = cand.split(), gold.split()
    wrong = [w for w in cw if w not in gw] or cw
    return any(not checker.is_word(w) for w in wrong)
