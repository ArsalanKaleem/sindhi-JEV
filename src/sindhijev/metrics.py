"""Evaluation metrics (no sklearn needed; checked against sklearn in tests when it is installed).

Inputs are per-candidate records:
  {"seq_y": 0/1, "seq_score": float, "char_y": [0/1...], "char_score": [float...], "ignore": [pos...],
   "groups": [...], "cluster": crop_id or example id, ...}
Higher score = more likely wrong.
"""
import math
import random
from collections import defaultdict


def auroc(y, s):
    """Mann-Whitney AUROC with tie handling. nan if one class is missing."""
    pairs = sorted(zip(s, y))
    n1 = sum(y)
    n0 = len(y) - n1
    if n0 == 0 or n1 == 0:
        return float("nan")
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2  # ranks i+1..j
        rank_sum += avg_rank * sum(p[1] for p in pairs[i:j])
        i = j
    return (rank_sum - n1 * (n1 + 1) / 2) / (n0 * n1)


def auprc(y, s):
    """Average precision (step-wise, as sklearn.average_precision_score)."""
    n1 = sum(y)
    if n1 == 0:
        return float("nan")
    order = sorted(range(len(y)), key=lambda i: -s[i])
    tp = fp = 0
    ap, prev_recall, i = 0.0, 0.0, 0
    while i < len(order):  # group ties
        j = i
        while j < len(order) and s[order[j]] == s[order[i]]:
            j += 1
        tp += sum(y[order[k]] for k in range(i, j))
        fp += sum(1 - y[order[k]] for k in range(i, j))
        recall = tp / n1
        ap += (recall - prev_recall) * (tp / (tp + fp))
        prev_recall = recall
        i = j
    return ap


def routing(y, s, fracs=(0.1, 0.2, 0.3)):
    """Flag the top f of candidates by score. Returns per f: recall of wrong candidates (errors caught),
    precision of the flagged set, and the error rate left in the unflagged (auto-accepted) part."""
    n = len(y)
    order = sorted(range(n), key=lambda i: -s[i])
    n1 = sum(y)
    out = {}
    for f in fracs:
        k = int(round(f * n))
        flagged = set(order[:k])
        caught = sum(y[i] for i in flagged)
        rest = n - k
        out[f"route@{int(f * 100)}"] = {
            "recall": caught / n1 if n1 else float("nan"),
            "precision": caught / k if k else float("nan"),
            "residual_error": (n1 - caught) / rest if rest else float("nan")}
    return out


def char_metrics(recs, thr):
    """Char-level P/R/F1 at threshold thr over all candidate positions (ignored positions skipped), and
    top-1 localisation: among wrong candidates with >=1 marked char, is the argmax char score a marked char?"""
    tp = fp = fn = 0
    hit = tot = 0
    for r in recs:
        ign = set(r.get("ignore", []))
        ys, ss = r["char_y"], r["char_score"]
        for k, (yy, sc) in enumerate(zip(ys, ss)):
            if k in ign:
                continue
            p = sc >= thr
            tp += p and yy == 1
            fp += p and yy == 0
            fn += (not p) and yy == 1
        cand = [k for k in range(len(ys)) if k not in ign]
        if r["seq_y"] == 1 and any(ys[k] for k in cand):
            tot += 1
            best = max(cand, key=lambda k: ss[k])
            hit += ys[best] == 1
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    return {"char_precision": prec, "char_recall": rec, "char_f1": f1,
            "top1_loc": hit / tot if tot else float("nan"), "n_loc": tot}


def best_char_threshold(recs):
    """Threshold maximising char F1 (chosen on the synthetic val set ONLY, then fixed)."""
    scores = sorted({round(s, 3) for r in recs for s in r["char_score"]})
    if len(scores) > 200:
        scores = [scores[int(i * (len(scores) - 1) / 199)] for i in range(200)]
    best = max(scores or [0.5], key=lambda t: char_metrics(recs, t)["char_f1"])
    return best


def summarize(recs, char_thr=None):
    y = [r["seq_y"] for r in recs]
    s = [r["seq_score"] for r in recs]
    out = {"n": len(recs), "n_wrong": sum(y), "auroc": auroc(y, s), "auprc": auprc(y, s),
           "base_rate": sum(y) / len(y) if y else float("nan")}
    out.update(routing(y, s))
    if char_thr is not None and recs and "char_score" in recs[0]:
        out.update(char_metrics(recs, char_thr))
    return out


def bootstrap_ci(recs, fn, n=1000, seed=0, cluster_key="cluster"):
    """95% CI of fn(recs) by resampling CLUSTERS (e.g. crops: one crop's 5 engine outputs move together)."""
    clusters = defaultdict(list)
    for r in recs:
        clusters[r.get(cluster_key, id(r))].append(r)
    keys = list(clusters)
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        sample = [r for k in (rng.choice(keys) for _ in keys) for r in clusters[k]]
        v = fn(sample)
        if not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    vals.sort()
    if not vals:
        return (float("nan"), float("nan"))
    return vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]


def paired_bootstrap_delta(recs, score_a, score_b, n=1000, seed=0, cluster_key="cluster"):
    """AUROC(a) - AUROC(b) on the same records, with a clustered bootstrap 95% CI and the share of resamples
    where a <= b (one-sided p-value style)."""
    def delta(rs):
        y = [r["seq_y"] for r in rs]
        return auroc(y, [r[score_a] for r in rs]) - auroc(y, [r[score_b] for r in rs])
    point = delta(recs)
    clusters = defaultdict(list)
    for r in recs:
        clusters[r.get(cluster_key, id(r))].append(r)
    keys = list(clusters)
    rng = random.Random(seed)
    ds = []
    for _ in range(n):
        d = delta([r for k in (rng.choice(keys) for _ in keys) for r in clusters[k]])
        if not math.isnan(d):
            ds.append(d)
    ds.sort()
    return {"delta": point, "ci": (ds[int(0.025 * (len(ds) - 1))], ds[int(0.975 * (len(ds) - 1))]),
            "p_a_not_better": sum(d <= 0 for d in ds) / len(ds)}


def breakdown(recs, key_fn, char_thr=None, min_n=1):
    """Group records by key_fn(rec) (may return a list of keys) and summarise each group."""
    groups = defaultdict(list)
    for r in recs:
        ks = key_fn(r)
        for k in (ks if isinstance(ks, (list, tuple, set)) else [ks]):
            groups[k].append(r)
    return {k: summarize(v, char_thr) for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))
            if len(v) >= min_n}


def ece(y, p, bins=10):
    """Expected calibration error of probabilities p for binary labels y (equal-width bins)."""
    n = len(y)
    if n == 0:
        return float("nan")
    tot = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i in range(n) if (lo <= p[i] < hi) or (b == bins - 1 and p[i] == 1.0)]
        if idx:
            conf = sum(p[i] for i in idx) / len(idx)
            acc = sum(y[i] for i in idx) / len(idx)
            tot += len(idx) / n * abs(conf - acc)
    return tot


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def selection(recs, score_key):
    """Per crop, keep the engine output with the LOWEST score_key (least likely wrong).
    Returns CER (sum of edits / sum of gold length) and exact-match rate of the kept outputs."""
    by = defaultdict(list)
    for r in recs:
        by[r["cluster"]].append(r)
    ed = ln = exact = 0
    for rs in by.values():
        best = min(rs, key=lambda r: (r[score_key], r["engine"]))
        ed += edit_distance(best["cand"], best["gold"])
        ln += max(1, len(best["gold"]))
        exact += best["cand"] == best["gold"]
    return {"cer": ed / ln if ln else float("nan"), "exact": exact / len(by) if by else float("nan"), "crops": len(by)}
