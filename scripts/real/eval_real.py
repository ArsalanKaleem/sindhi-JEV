"""FINAL evaluation on the frozen real set. Run once per reported model, after all choices are fixed.

    python scripts/real/eval_real.py --final --ckpt runs/main_v2/last.pt --name main_v2 --root data/real/addendum_v1 \
        --extra abl_ref=runs/abl_ref/last.pt ... --jev runs/jev_real/jev_words.jsonl

--extra NAME=CKPT adds another checkpoint as a method (ablations, v1); its char threshold comes from
results/synth_NAME.json. --jev adds MiniCPM-OCR Jev scores (rows {"id": "crop_id|engine", "prob_1"}; error score
1 - prob_1; failed items scored 0.5 and counted). All methods are scored in ONE run per root.

Refuses to run without --final or if the set changed since freeze (realset.load_real). Every run is
appended to docs/real_set_access_log.md.

Methods compared on the same (crop, engine) candidates:
  model   : the verifier (candidate re-rendered in the candidate font)
  ocr_conf: (a) 1 - engine confidence
  text    : (b) char n-gram LM + dictionary (results/text_checker.pkl, fitted on synthetic val only)
  voting  : (c) disagreement of the other engines on the same crop
Main labels exclude encoding-variant errors (decision 2026-11-17: ه/ھ/ہ, ي/ی/ى, ڪ/ک/ك reported separately);
ambiguous gold positions are excluded from char metrics. Thresholds: model/text char thresholds come from
synthetic val; ocr_conf/voting use 0.5 (no tuning on real data). Top-1 localisation needs no threshold.
"""
import argparse
import datetime
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from sindhijev import realset as R  # noqa: E402
from sindhijev.baselines import TextChecker, is_nonword_error, ocr_confidence, voting  # noqa: E402
from sindhijev.data import RealSet, collate  # noqa: E402
from sindhijev.metrics import (auroc, bootstrap_ci, char_metrics, ece, edit_distance,  # noqa: E402
                               paired_bootstrap_delta, selection, summarize)
from sindhijev.render import Renderer  # noqa: E402
from evaluate import load_model  # noqa: E402
from train import predict  # noqa: E402

BASELINES = ["ocr_conf", "text", "voting"]


def set_name(root):
    """words / lines / words_addendum / lines_addendum (both addenda are folders named addendum_v1)."""
    parts = Path(root).resolve().parts
    base = "lines" if "lines" in parts else ("words" if "real" in parts else Path(root).name)
    return base + ("_addendum" if "addendum_v1" in parts else "")


def encoding_groups():
    data = json.loads((ROOT / "data" / "confusion_set.json").read_text(encoding="utf-8"))
    return {g["id"] for g in data["groups"] if g["kind"] == "encoding"}


def main_labels(rec, enc):
    """Drop encoding-variant substitutions from the labels (reported separately)."""
    y = list(rec["char_y_all"])
    enc_only = False
    errs = rec["errors"]
    for e in errs:
        if e.get("op") == "sub" and e.get("group") in enc and 0 <= e["cand_pos"] < len(y):
            y[e["cand_pos"]] = 0
    seq = int(any(y) or (not rec["cand"] and bool(rec["gold"])))
    if rec["seq_y_all"] and not seq:
        enc_only = True
    return seq, y, enc_only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--root", default=str(ROOT / "data" / "real"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--extra", action="append", default=[], help="NAME=CKPT, repeatable")
    ap.add_argument("--jev", default=None, help="Jev scores jsonl for this root")
    a = ap.parse_args()
    extras = dict(x.split("=", 1) for x in a.extra)
    root = Path(a.root)
    set_tag = set_name(root)
    rows = R.load_real(final=a.final, root=root)  # raises unless --final and hashes match FROZEN.json
    log = ROOT / "docs" / "real_set_access_log.md"
    real_root = (ROOT / "data" / "real").resolve()
    is_real = root.resolve() == real_root or real_root in root.resolve().parents
    with open(log if is_real else Path.home() / ".sindhijev_test_access.log", "a", encoding="utf-8") as f:
        f.write(f"- {datetime.datetime.now():%Y-%m-%d %H:%M} eval_real --final name={a.name} ckpt={a.ckpt} "
                f"root={set_tag} extras={sorted(extras)} jev={bool(a.jev)}\n")

    dev = torch.device(a.device)
    model, st = load_model(a.ckpt, dev)
    cands = R.read_csv(root / "candidates.csv")
    crops = R.read_csv(root / "crops.csv")
    rend = Renderer(ROOT / "fonts" / model.cfg.get("candidate_font", "NotoNaskhArabic.ttf"), size=48)
    ds = RealSet(root, rows, cands, crops, model.cfg, rend)
    recs = predict(model, DataLoader(ds, batch_size=64, collate_fn=collate), dev, False)
    extra_recs = {}
    for name, ck in extras.items():
        m_x, _ = load_model(ck, dev)
        extra_recs[name] = predict(m_x, DataLoader(RealSet(root, rows, cands, crops, m_x.cfg, rend), batch_size=64,
                                                   collate_fn=collate), dev, False)
        print("scored", name, flush=True)
    jev, jev_failed = None, 0
    if a.jev:
        jev = {}
        for x in Path(a.jev).read_text(encoding="utf-8").splitlines():
            j = json.loads(x)
            ok = j.get("prob_1") is not None and not j.get("error") and 0.0 <= j["prob_1"] <= 1.0
            jev_failed += not ok
            jev[j["id"]] = 1 - j["prob_1"] if ok else 0.5
        missing = [f"{r['crop_id']}|{r['engine']}" for r in ds.rows if f"{r['crop_id']}|{r['engine']}" not in jev]
        assert not missing, f"Jev scores missing for {len(missing)} candidates, e.g. {missing[:3]}"
    METHODS = ["model"] + list(extras) + (["jev"] if jev is not None else []) + BASELINES
    no_char = {"jev"} | {n for n in extras if "nocharhead" in n}  # no per-letter output: char metrics n/a

    tc = TextChecker.load(ROOT / "results" / "text_checker.pkl")
    synth = json.loads((ROOT / "results" / f"synth_{a.name}.json").read_text(encoding="utf-8"))
    text_cfg = json.loads((ROOT / "results" / "synth_text_baseline.json").read_text(encoding="utf-8"))
    thr = {"model": synth["char_threshold_from_val"], "text": text_cfg["char_threshold_from_val"],
           "ocr_conf": 0.5, "voting": 0.5, "jev": 0.5}
    for name in extras:
        thr[name] = json.loads((ROOT / "results" / f"synth_{name}.json").read_text(encoding="utf-8"))[
            "char_threshold_from_val"]
    enc = encoding_groups()
    by_crop = {}
    for r in ds.rows:
        by_crop.setdefault(r["crop_id"], {})[r["engine"]] = r["cand_text"]
    out_recs = []
    for i, (r, row) in enumerate(zip(recs, ds.rows)):
        cand = row["cand_text"]
        rec = {"cluster": row["crop_id"], "engine": row["engine"], "stratum": row["stratum"],
               "source": row["source_id"], "cand": cand, "gold": row["gold_text"],
               "seq_y_all": int(row["seq_label"]), "char_y_all": json.loads(row["char_labels_json"]),
               "errors": json.loads(row["errors_json"]), "ignore": json.loads(row["ambiguous_cand_json"] or "[]")}
        rec["seq_y"], rec["char_y"], rec["encoding_only"] = main_labels(rec, enc)
        n = len(cand)
        s = {"model": (r["seq_score"], r["char_score"][:n]),
             "ocr_conf": ocr_confidence(cand, row["seq_conf"], row["char_conf_json"]),
             "text": tc.score(cand),
             "voting": voting(cand, [t for e, t in by_crop[row["crop_id"]].items() if e != row["engine"]])}
        for name, xr in extra_recs.items():
            s[name] = (xr[i]["seq_score"], xr[i]["char_score"][:n])
        if jev is not None:
            s["jev"] = (jev[f"{row['crop_id']}|{row['engine']}"], [0.0] * n)
        for m, (sq, ch) in s.items():
            rec[f"{m}_seq"] = sq
            rec[f"{m}_char"] = (list(ch) + [0.0] * n)[:n]
        rec["error_kind"] = ("correct" if not rec["seq_y"] else
                             "nonword" if is_nonword_error(cand, rec["gold"], tc) else "realword")
        rec["dot_error"] = any(e.get("group") not in (None, "other") and e.get("group") not in enc
                               for e in rec["errors"])
        out_recs.append(rec)

    def view(rs, m):
        return [{**x, "seq_score": x[f"{m}_seq"], "char_score": x[f"{m}_char"]} for x in rs]

    def table(rs):
        t = {}
        for m in METHODS:
            v = view(rs, m)
            o = summarize(v, thr[m])
            if m in no_char:
                for k in [k for k in o if k.startswith("char_") or k in ("top1_loc", "n_loc")]:
                    o[k] = None
            o["auroc_ci"] = bootstrap_ci(v, lambda q: auroc([x["seq_y"] for x in q], [x["seq_score"] for x in q]),
                                         n=a.boot)
            t[m] = o
        return t

    res = {"name": a.name, "ckpt": a.ckpt, "extras": extras, "jev_file": a.jev, "jev_failed": jev_failed,
           "methods": METHODS, "n": len(out_recs), "thresholds": thr,
           "encoding_only_wrong": dict(Counter(r["engine"] for r in out_recs if r["encoding_only"])),
           "all": table(out_recs)}
    res["by_engine"] = {e: table([r for r in out_recs if r["engine"] == e]) for e in sorted({r["engine"] for r in out_recs})}
    res["by_stratum"] = {s: table([r for r in out_recs if r["stratum"] == s]) for s in sorted({r["stratum"] for r in out_recs})}
    correct = [r for r in out_recs if not r["seq_y"]]
    for kind in ("nonword", "realword"):
        sub = correct + [r for r in out_recs if r["error_kind"] == kind]
        res[f"correct_vs_{kind}"] = table(sub) if any(r["seq_y"] for r in sub) else None
    sub = correct + [r for r in out_recs if r["dot_error"]]
    res["correct_vs_dot_errors"] = table(sub)
    res["paired_vs_model"] = {m: paired_bootstrap_delta([{**x, "a": x["model_seq"], "b": x[f"{m}_seq"]}
                                                          for x in out_recs], "a", "b", n=a.boot)
                              for m in METHODS[1:]}
    res["char_by_method_dot_errors_only"] = {m: char_metrics(view([r for r in out_recs if r["dot_error"]], m), thr[m])
                                             for m in METHODS if m not in no_char}
    # Engine selection: per crop keep the output a method trusts most; compare with each single engine and the
    # oracle (best possible pick). Uses ALL errors (incl. encoding variants) because this measures final text.
    sel = {f"select_by_{m}": selection(out_recs, f"{m}_seq") for m in METHODS}
    for e in sorted({r["engine"] for r in out_recs}):
        sel[f"always_{e}"] = selection([{**r, "_k": 0} for r in out_recs if r["engine"] == e], "_k")
    sel["oracle"] = selection([{**r, "_k": edit_distance(r["cand"], r["gold"])} for r in out_recs], "_k")
    res["engine_selection"] = sel
    # calibration of the model's probability on real data (reported, never tuned on it)
    res["model_ece"] = ece([r["seq_y"] for r in out_recs], [r["model_seq"] for r in out_recs])
    od = ROOT / "results"
    od.mkdir(exist_ok=True)
    tag = f"{a.name}_{set_tag}"
    (od / f"real_{tag}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    f3 = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
    lines = [f"# Real-set results: {a.name} on {set_tag} ({len(out_recs)} candidates, "
             f"{len({r['cluster'] for r in out_recs})} crops)", "",
             "| method | AUROC [95% CI] | AUPRC | char F1 | top-1 loc | recall@10% | recall@20% | recall@30% |",
             "|---|---|---|---|---|---|---|---|"]
    for m in METHODS:
        o = res["all"][m]
        lines.append(f"| {m} | {o['auroc']:.3f} [{o['auroc_ci'][0]:.3f}, {o['auroc_ci'][1]:.3f}] | {o['auprc']:.3f} | "
                     f"{f3(o.get('char_f1'))} | {f3(o.get('top1_loc'))} | "
                     f"{o['route@10']['recall']:.3f} | {o['route@20']['recall']:.3f} | {o['route@30']['recall']:.3f} |")
    lines += ["", "Model minus baseline, AUROC (paired, crop-clustered bootstrap):", ""]
    for m, d in res["paired_vs_model"].items():
        lines.append(f"- vs {m}: {d['delta']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}], "
                     f"share of resamples where model is not better: {d['p_a_not_better']:.3f}")
    lines += ["", "Engine selection (per crop, keep the output the method trusts most):", "",
              "| strategy | CER | exact match |", "|---|---|---|"]
    for k, v in sel.items():
        lines.append(f"| {k} | {v['cer']:.3f} | {v['exact']:.3f} |")
    lines += ["", f"Model ECE on real data: {res['model_ece']:.3f}"]
    (od / f"real_{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
