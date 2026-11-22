"""Phase 2 commands for the real test set.

    python scripts/real/realset_cli.py derive    [--pass 1]      # gold.csv + candidates.csv -> candidate_labels.csv
    python scripts/real/realset_cli.py freeze                     # write FROZEN.json (hashes of all files)
    python scripts/real/realset_cli.py verify                     # check nothing changed since freeze
    python scripts/real/realset_cli.py relabel-sample [--frac 0.1]  # -> relabel_queue.txt (stratified, seeded)
    python scripts/real/realset_cli.py agreement --pass2 gold_pass2.csv   # inter-pass consistency report
    python scripts/real/realset_cli.py summary                    # counts only (no per-item browsing)

gold.csv holds pass-1 labels (pass 2 lives in its own file and never overwrites pass 1).
"""
import argparse
import collections
import csv
import datetime
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sindhijev import realset as R  # noqa: E402

OUT_FIELDS = ["crop_id", "engine", "cand_text", "gold_text", "seq_label", "char_labels_json", "errors_json",
              "ambiguous_cand_json", "overridden", "seq_conf"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["derive", "freeze", "verify", "relabel-sample", "agreement", "summary"])
    ap.add_argument("--root", default=str(R.REAL))
    ap.add_argument("--pass", dest="pass_", default="1")
    ap.add_argument("--pass2", default=None)
    ap.add_argument("--frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--ignore-overrides", action="store_true",
                    help="drop override_json (pass-1 overrides 2026-11-17 were accidental clicks)")
    a = ap.parse_args()
    root = Path(a.root)

    if a.cmd == "derive":
        cands = R.read_csv(root / "candidates.csv")
        gold = [g for g in R.read_csv(root / "gold.csv") if g["pass"] == a.pass_]
        empty = [g["crop_id"] for g in gold if g["status"] == "ok" and not R.norm(g["gold_text"])]
        if empty:  # an 'ok' crop with no text would mark every engine wrong: skip it and say so
            print(f"WARNING: skipped {len(empty)} ok crop(s) with empty gold text: {empty}")
            gold = [g for g in gold if g["crop_id"] not in empty]
        rows = R.derive_all(cands, gold, ignore_overrides=a.ignore_overrides)
        with open(root / "candidate_labels.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, OUT_FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"{len(rows)} candidate labels from {len(gold)} gold rows")

    elif a.cmd == "freeze":
        if (root / "FROZEN.json").exists():
            sys.exit("FROZEN.json exists. Re-freezing changes the test set: delete it by hand and log why.")
        data = {"frozen_at": datetime.datetime.now().isoformat(timespec="seconds"), "files": R.file_hashes(root)}
        (root / "FROZEN.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
        print(f"froze {len(data['files'])} files")

    elif a.cmd == "verify":
        changed = R.check_frozen(root)
        print("OK: unchanged since freeze" if not changed else f"CHANGED: {changed}")
        sys.exit(1 if changed else 0)

    elif a.cmd == "relabel-sample":
        crops = {c["crop_id"]: c for c in R.read_csv(root / "crops.csv")}
        gold = [g for g in R.read_csv(root / "gold.csv") if g["pass"] == "1" and g["status"] == "ok"]
        by = collections.defaultdict(list)
        for g in gold:
            by[crops.get(g["crop_id"], {}).get("stratum", "?")].append(g["crop_id"])
        rng = random.Random(a.seed)
        pick = []
        for s, ids in sorted(by.items()):
            ids = sorted(ids)
            pick += rng.sample(ids, max(1, round(a.frac * len(ids))))
        (root / "relabel_queue.txt").write_text("\n".join(sorted(pick)) + "\n", encoding="utf-8")
        print(f"{len(pick)} crops queued for blind pass 2 -> relabel_queue.txt")

    elif a.cmd == "agreement":
        cands = R.read_csv(root / "candidates.csv")
        p1 = [g for g in R.read_csv(root / "gold.csv") if g["pass"] == "1"]
        p2 = R.read_csv(a.pass2)
        rep = R.agreement(cands, p1, p2)
        (root / "agreement.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(json.dumps(rep, indent=1))

    elif a.cmd == "summary":
        crops = R.read_csv(root / "crops.csv")
        gold = [g for g in R.read_csv(root / "gold.csv") if g["pass"] == "1"] if (root / "gold.csv").exists() else []
        print("crops by stratum:", dict(collections.Counter(c["stratum"] for c in crops)))
        print("gold by status:  ", dict(collections.Counter(g["status"] for g in gold)))
        secs = [int(g["seconds"] or 0) for g in gold]
        if secs:
            print(f"labelling time: {sum(secs) / 3600:.1f} h total, {sum(secs) / len(secs):.0f} s/crop")


if __name__ == "__main__":
    main()
