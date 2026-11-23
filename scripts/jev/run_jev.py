"""Client for the MiniCPM-OCR Jev score-only service (/v1/score), run on Kaggle next to the service (stdlib only).

    python scripts/jev/run_jev.py --set data/synth/test_hard --out jev_test_hard.jsonl [--limit 50] [--max-hours 1]
    python scripts/jev/run_jev.py --items items.jsonl --out jev_real.jsonl      # final evaluation (crops supplied then)

We supply the render ourselves (score-only path): Jev's own renderer has no RTL handling and installs CJK fonts only
(docs/decisions.md, 2026-11-24). Synthetic sets: original_image = scan/<id>.png, render_image = cand/<id>.png
(our HarfBuzz render of the candidate), ocr_text = cand. --items rows: {"id", "original_image", "render_image",
"ocr_text"} with paths relative to the items file.
Output rows: {"id", "prob_1", "error", "sec"}; prob_1 = Jev's P(candidate correct). Resumable: done ids are skipped.
"""
import argparse
import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def load_items(a):
    if a.set:
        d = Path(a.set)
        with open(d / "manifest.jsonl", encoding="utf-8") as f:
            rows = [json.loads(x) for x in f]
        return [{"id": r["id"], "original_image": d / "scan" / f"{r['id']}.png",
                 "render_image": d / "cand" / f"{r['id']}.png", "ocr_text": r["cand"]} for r in rows]
    base = Path(a.items).parent
    with open(a.items, encoding="utf-8") as f:
        rows = [json.loads(x) for x in f]
    return [{**r, "original_image": base / r["original_image"], "render_image": base / r["render_image"]} for r in rows]


def b64(p):
    return base64.b64encode(Path(p).read_bytes()).decode()


def post(url, body, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--set")
    g.add_argument("--items")
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:18501/v1/score")
    ap.add_argument("--batch", type=int, default=16, help="items per request (service cap 64)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-hours", type=float, default=None)
    a = ap.parse_args()
    items = load_items(a)[:a.limit]
    out = Path(a.out)
    done = set()
    if out.exists():
        done = {json.loads(x)["id"] for x in out.read_text(encoding="utf-8").splitlines() if x.strip()}
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(items)} items, {len(done)} done, {len(todo)} to score", flush=True)
    t0 = time.time()
    with open(out, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), a.batch):
            if a.max_hours and (time.time() - t0) / 3600 > a.max_hours:
                print("time budget reached; rerun to resume", flush=True)
                break
            chunk = todo[i:i + a.batch]
            body = {"items": [{"original_image": b64(it["original_image"]), "render_image": b64(it["render_image"]),
                               "ocr_text": it["ocr_text"]} for it in chunk]}
            t = time.time()
            try:
                res = post(a.url, body)["results"]
            except (urllib.error.URLError, TimeoutError, KeyError) as e:
                res = [{"prob_1": None, "error": f"request: {e}"}] * len(chunk)
            sec = (time.time() - t) / len(chunk)
            for it, r in zip(chunk, res):
                f.write(json.dumps({"id": it["id"], "prob_1": r.get("prob_1"), "error": r.get("error"),
                                    "sec": round(sec, 4)}, ensure_ascii=False) + "\n")
            f.flush()
            n = i + len(chunk)
            if n % (a.batch * 10) == 0 or n == len(todo):
                el = time.time() - t0
                print(f"{n}/{len(todo)}  {el / n:.3f} s/item  ETA {el / n * (len(todo) - n) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
