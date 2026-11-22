"""Real test set: loading behind a freeze guard, label derivation, agreement.

The real set must stay untouched until final evaluation. `load_real()` raises
unless called with final=True AND the files match data/real/FROZEN.json.
"""
import csv
import hashlib
import json
import math
import unicodedata
from pathlib import Path

from .align import derive_labels, levenshtein_ops, load_pair_groups

ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / "data" / "real"
FROZEN_FILES = ["crops.csv", "candidates.csv", "gold.csv", "sources.csv", "candidate_labels.csv"]


PUNCT_MAP = str.maketrans({",": "،", ";": "؛", "?": "؟"})  # Latin vs Arabic punctuation: same mark, not an OCR error


def norm(s):
    """NFC, drop ZWNJ/ZWJ, collapse spaces, Latin , ; ? -> Arabic (applied to gold AND candidates)."""
    return " ".join(unicodedata.normalize("NFC", s or "").replace("‌", "").replace("‍", "")
                    .translate(PUNCT_MAP).split())


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def file_hashes(root=REAL):
    root = Path(root)
    out = {n: sha256(root / n) for n in FROZEN_FILES if (root / n).exists()}
    for p in sorted((root / "crops").glob("*")):
        out[f"crops/{p.name}"] = sha256(p)
    return out


def check_frozen(root=REAL):
    frozen = json.loads((Path(root) / "FROZEN.json").read_text(encoding="utf-8"))
    now = file_hashes(root)
    changed = sorted(k for k in set(frozen["files"]) | set(now) if frozen["files"].get(k) != now.get(k))
    return changed


def load_real(final=False, root=REAL):
    if not final:
        raise PermissionError("The real test set is for FINAL evaluation only. Pass final=True (--final) "
                              "and log the run in docs/real_set_access_log.md.")
    changed = check_frozen(root)
    if changed:
        raise RuntimeError(f"Real set differs from FROZEN.json: {changed[:5]} ...")
    return read_csv(Path(root) / "candidate_labels.csv")


def derive_all(candidates, gold_rows, pair_groups=None, ignore_overrides=False):
    """candidates: rows of candidates.csv; gold_rows: rows of gold.csv for ONE pass."""
    pair_groups = pair_groups if pair_groups is not None else load_pair_groups()
    gold = {g["crop_id"]: g for g in gold_rows}
    out = []
    for c in candidates:
        g = gold.get(c["crop_id"])
        if not g or g["status"] != "ok":
            continue
        gt, ct = norm(g["gold_text"]), norm(c["cand_text"])
        d = derive_labels(gt, ct, pair_groups)
        labels = d["char_labels"]
        ovr = (json.loads(g["override_json"]).get(c["engine"], [])
               if g.get("override_json") and not ignore_overrides else [])
        for k in ovr:
            if 0 <= k < len(labels):
                labels[k] = 1 - labels[k]
        # gold positions the labeller marked ambiguous -> aligned candidate positions (excluded from char metrics)
        amb_gold = {int(x) for x in (g.get("ambiguous_chars") or "").split()}
        amb_cand = sorted(cj for op, gi, cj in levenshtein_ops(gt, ct) if op in ("eq", "sub") and gi in amb_gold)
        seq = int(any(labels) or (not ct and bool(gt)))
        out.append({"crop_id": c["crop_id"], "engine": c["engine"], "cand_text": ct, "gold_text": gt,
                    "seq_label": seq, "char_labels_json": json.dumps(labels),
                    "errors_json": json.dumps(d["errors"], ensure_ascii=False),
                    "ambiguous_cand_json": json.dumps(amb_cand), "overridden": "Y" if ovr else "N",
                    "seq_conf": c.get("seq_conf", "")})
    return out


def cohen_kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def cer(ref, hyp):
    if not ref:
        return float(len(hyp) > 0)
    return sum(op != "eq" for op, _, _ in levenshtein_ops(ref, hyp)) / len(ref)


def agreement(cands, pass1, pass2, pair_groups=None):
    """Inter-pass consistency on the crops labelled in both passes."""
    p1 = {g["crop_id"]: g for g in pass1}
    p2 = {g["crop_id"]: g for g in pass2}
    both = sorted(set(p1) & set(p2))
    status_agree = sum(p1[c]["status"] == p2[c]["status"] for c in both)
    ok = [c for c in both if p1[c]["status"] == p2[c]["status"] == "ok"]
    exact = sum(norm(p1[c]["gold_text"]) == norm(p2[c]["gold_text"]) for c in ok)
    cers = [cer(norm(p1[c]["gold_text"]), norm(p2[c]["gold_text"])) for c in ok]
    sub = [c for c in cands if c["crop_id"] in set(ok)]
    d1 = {(r["crop_id"], r["engine"]): r for r in derive_all(sub, [p1[c] for c in ok], pair_groups)}
    d2 = {(r["crop_id"], r["engine"]): r for r in derive_all(sub, [p2[c] for c in ok], pair_groups)}
    keys = sorted(set(d1) & set(d2))
    s1 = [d1[k]["seq_label"] for k in keys]
    s2 = [d2[k]["seq_label"] for k in keys]
    tp = fp = fn = 0
    for k in keys:
        for x, y in zip(json.loads(d1[k]["char_labels_json"]), json.loads(d2[k]["char_labels_json"])):
            tp += x and y
            fp += y and not x
            fn += x and not y
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    return {"crops_both_passes": len(both), "status_agreement": status_agree / len(both) if both else math.nan,
            "crops_ok_both": len(ok), "gold_exact_match": exact / len(ok) if ok else math.nan,
            "gold_mean_cer": sum(cers) / len(cers) if cers else math.nan,
            "candidates_compared": len(keys), "seq_label_kappa": cohen_kappa(s1, s2),
            "seq_label_agreement": sum(x == y for x, y in zip(s1, s2)) / len(keys) if keys else math.nan,
            "char_mark_f1_pass2_vs_pass1": f1}
