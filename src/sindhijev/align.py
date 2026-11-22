"""Derive candidate labels from a gold transcription (real test set).

Same rules as the synthetic data (corrupt.py), per code point of the CANDIDATE:
  substituted char      -> 1
  inserted char         -> 1
  deleted gold char     -> the candidate char right after the gap = 1
                           (right before it if the gap is at the end)
Alignment: Levenshtein over code points, ties broken toward substitution, so
a dot swap (ب -> ٻ) is one substitution, not a deletion + insertion.

Each error is also tagged with its confusion group when the (gold, cand)
pair is in data/confusion_set.json (any kind), else 'other'.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def levenshtein_ops(gold, cand):
    """Return list of (op, gold_i, cand_j) with op in {'eq','sub','del','ins'}."""
    n, m = len(gold), len(cand)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j - 1] + (gold[i - 1] != cand[j - 1]), D[i - 1][j] + 1, D[i][j - 1] + 1)
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (gold[i - 1] != cand[j - 1]):
            ops.append(("eq" if gold[i - 1] == cand[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            ops.append(("del", i - 1, j))
            i -= 1
        else:
            ops.append(("ins", i, j - 1))
            j -= 1
    return ops[::-1]


def load_pair_groups(path=None):
    data = json.load(open(path or ROOT / "data" / "confusion_set.json", encoding="utf-8"))
    pairs = {}
    for g in data["groups"]:
        cs = [m["c"] for m in g["members"]]
        if "pairs" in g and g["kind"] != "encoding":  # v1: reviewed pairs only; rejected pairs -> 'other'
            ok = [(p["a"], p["b"]) for p in g["pairs"]]
        else:  # encoding variants are code-point facts, tagged whatever their visual similarity
            ok = [(a, b) for a in cs for b in cs if a != b]
        for a, b in ok:
            pairs.setdefault((a, b), g["id"])
            pairs.setdefault((b, a), g["id"])
    return pairs


def derive_labels(gold, cand, pair_groups=None):
    """Return dict(seq_label, char_labels, errors[list of dicts])."""
    pair_groups = pair_groups if pair_groups is not None else load_pair_groups()
    labels = [0] * len(cand)
    errors = []
    for op, gi, cj in levenshtein_ops(gold, cand):
        if op == "eq":
            continue
        if op == "sub":
            labels[cj] = 1
            errors.append({"op": "sub", "cand_pos": cj, "gold": gold[gi], "cand": cand[cj],
                           "group": pair_groups.get((gold[gi], cand[cj]), "other")})
        elif op == "ins":
            labels[cj] = 1
            errors.append({"op": "ins", "cand_pos": cj, "cand": cand[cj], "group": "other"})
        else:  # deletion: gold char missing before candidate position cj
            pos = cj if cj < len(cand) else len(cand) - 1
            if pos >= 0:
                labels[pos] = 1
            errors.append({"op": "del", "cand_pos": pos, "gap": cj, "gold": gold[gi], "group": "other"})
    # Digraph swaps (ج <-> جه) align as an insertion/deletion of ه right after ج / گ; tag them.
    for e in errors:
        if e["op"] in ("ins", "del") and (e.get("cand") or e.get("gold")) == "ه":
            p = e["cand_pos"] if e["op"] == "ins" else e["gap"]
            prev = cand[p - 1] if p > 0 else ""
            if prev in ("ج", "گ"):
                e["group"] = "jeem_digraph" if prev == "ج" else "gaf_digraph"
    return {"seq_label": int(bool(errors)), "char_labels": labels, "errors": errors}
