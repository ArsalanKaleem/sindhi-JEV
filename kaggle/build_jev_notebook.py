"""Generate kaggle/jev.ipynb (MiniCPM-OCR Jev zero-shot on synthetic test sets, T4, <= 2 GPU h)."""
import json
from pathlib import Path

CELLS = [
    ("markdown", """# Sindhi-Jev: MiniCPM-OCR Jev zero-shot baseline

Accelerator **GPU T4 x1**, **Internet ON**. Input dataset: `sindhijev-jev` (MODE "synthetic",
`kaggle/make_jev_bundle.py`) or the PRIVATE `sindhijev-jev-real` (MODE "real", `scripts/jev/make_real_items.py`).
First run with `DEBUG = True` (about 15 min: install, self-check, 50 items per set, speed). Then `DEBUG = False`
with **Save & Run All (commit)**, and download `jev_results.zip` from the Output tab."""),
    ("code", """# Settings
MODE = "synthetic"                # "synthetic" (test_hard, test_font_hard) or "real" (frozen real candidates, final eval)
DEBUG = True
DTYPES = ["float16", "float32"]   # T4 has no bf16 (Jev's default); first dtype that passes the self-check is used
MAX_HOURS = 1.5                   # scoring time cap for this session (baseline budget: <= 2 GPU h in total)
SETS = ["test_hard", "test_font_hard"] if MODE == "synthetic" else ["words", "lines"]
JEV_COMMIT = "c21a3ca524263ef8166cb6d350864b10c3056b61"   # same code and weights as the synthetic run (2026-11-25)
WEIGHTS_REV = "a0ed278eb6aa1f333a711665ab610055a4a816ab"
PORT = 18501
SELFCHECK_TOL = 0.02              # |our prob - OpenBMB reference prob| allowed on their bundled examples"""),
    ("code", """import glob, os, sys, json, time, subprocess
T0 = time.time()
marker = "data/synth/test_hard/manifest.jsonl" if MODE == "synthetic" else "real_words/items.jsonl"
hits = glob.glob(f"/kaggle/input/**/{marker}", recursive=True)
assert hits, f"attach the {'sindhijev-jev' if MODE == 'synthetic' else 'sindhijev-jev-real'} dataset"
SRC = hits[0][: -len(marker) - 1]
assert os.path.exists(f"{SRC}/scripts/jev/run_jev.py"), "run_jev.py missing from the bundle"
OUT = "/kaggle/working/jev"
os.makedirs(OUT, exist_ok=True)
print("bundle:", SRC)
for s in SETS:  # fail here (before any GPU work) if the bundle layout is not what run_jev.py expects
    if MODE == "synthetic":
        for sub in ("manifest.jsonl", "scan", "cand"):
            p = f"{SRC}/data/synth/{s}/{sub}"
            assert os.path.exists(p), f"missing {p}; the dataset layout differs: !ls -R /kaggle/input | head"
        print(s, len(os.listdir(f"{SRC}/data/synth/{s}/scan")), "scan images")
    else:
        rows = [json.loads(x) for x in open(f"{SRC}/real_{s}/items.jsonl", encoding="utf-8")]
        miss = [r["id"] for r in rows if not (os.path.exists(f"{SRC}/real_{s}/{r['original_image']}")
                                             and os.path.exists(f"{SRC}/real_{s}/{r['render_image']}"))]
        assert not miss, f"{len(miss)} items with missing images, e.g. {miss[:3]}"
        print(s, len(rows), "items, all images present")
!nvidia-smi --query-gpu=name,memory.total --format=csv"""),
    ("code", """# Install Jev SDK (transformers backend) and download the weights
REPO = "/kaggle/working/MiniCPM-OCR"
if not os.path.exists(REPO):
    !git clone -q https://github.com/OpenBMB/MiniCPM-OCR {REPO}
subprocess.run(["git", "-C", REPO, "checkout", "-q", JEV_COMMIT], check=True)
COMMIT = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
JEV = f"{REPO}/scripts/minicpm-ocr-jev"
!pip install -q -r {JEV}/sdk/requirements-transformers.txt
# Kaggle's preinstalled torchaudio is built for another CUDA than the new torch, and transformers then fails to import
# AutoProcessor (seen 2026-11-24). Jev does not use audio.
!pip uninstall -y -q torchaudio
!python3 -c "from transformers import AutoProcessor, Qwen3_5ForSequenceClassification; print('import ok')"
from huggingface_hub import snapshot_download
WEIGHTS = snapshot_download("openbmb/MiniCPM-OCR-Jev", revision=WEIGHTS_REV)
import torch, transformers
print("commit", COMMIT, "| torch", torch.__version__, "| transformers", transformers.__version__)"""),
    ("code", """# Start the score-only service; keep the first dtype whose self-check matches OpenBMB's reference examples
import urllib.request

def healthz():
    try:
        return json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/healthz", timeout=5))
    except Exception as e:
        return {"status": f"down ({e.__class__.__name__})"}

def start(dtype):
    cfg = f"{OUT}/transformers_{dtype}.yaml"
    open(cfg, "w").write(f"engine: transformers\\nmodel_path: {WEIGHTS}\\ndevice: auto\\ndtype: {dtype}\\n"
                         f"attn_implementation: sdpa\\nmax_in_flight: 1\\nmax_batch_size: 8\\n")
    log = open(f"{OUT}/service_{dtype}.log", "w")
    p = subprocess.Popen(["bash", "sdk/start_score.sh", str(PORT)], cwd=JEV, stdout=log, stderr=subprocess.STDOUT,
                         env={**os.environ, "JEV_BACKEND_CONFIG": cfg, "JEV_WEIGHTS": WEIGHTS})  # full prompt self-check
    for _ in range(120):
        time.sleep(5)
        h = healthz()
        if h.get("status") == "ok" or p.poll() is not None:
            break
    return p, h

def reference_check():
    import base64
    diffs = {}
    for case in sorted(glob.glob(f"{JEV}/examples/*/case.json")):
        d = os.path.dirname(case); c = json.load(open(case))
        role = c["reference_render_of"]
        text = next(x["text"] for x in c["candidates"] if x["role"] == role)
        b64 = lambda f: base64.b64encode(open(f"{d}/{f}", "rb").read()).decode()
        body = {"items": [{"original_image": b64(c["block_image"]), "render_image": b64(c["reference_render"]),
                           "ocr_text": text}]}
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/score", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        got = json.load(urllib.request.urlopen(req, timeout=300))["results"][0]["prob_1"]
        diffs[os.path.basename(d)] = (got, c["observed_prob_yes"][role])
    return diffs

DTYPE, proc, checks = None, None, {}
for dt in DTYPES:
    proc, h = start(dt)
    diffs = reference_check() if h.get("status") == "ok" else {}
    ok = bool(diffs) and all(g is not None and abs(g - r) <= SELFCHECK_TOL for g, r in diffs.values())
    checks[dt] = {"healthz": h, "reference": diffs, "pass": ok}
    print(dt, json.dumps(checks[dt], default=str)[:600])
    if ok:
        DTYPE = dt
        break
    proc.terminate(); proc.wait()
assert DTYPE, "no dtype passed the self-check: see service_*.log; do not run the scoring cell"
print("using", DTYPE)"""),
    ("code", """# Score the chosen sets (resumable; per-session time cap). Any client failure stops the notebook.
for s in SETS:
    left = MAX_HOURS - (time.time() - T0) / 3600
    if left <= 0:
        print("time cap reached before", s)
        break
    src = ["--set", f"{SRC}/data/synth/{s}"] if MODE == "synthetic" else ["--items", f"{SRC}/real_{s}/items.jsonl"]
    cmd = [sys.executable, f"{SRC}/scripts/jev/run_jev.py", *src,
           "--out", f"{OUT}/jev_{s}.jsonl", "--batch", "16", "--max-hours", f"{left:.3f}"]
    if DEBUG:
        cmd += ["--limit", "50"]
    print(" ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
    assert r.returncode == 0, f"run_jev.py failed on {s} (exit {r.returncode}); see the error above"
    n = sum(1 for _ in open(f"{OUT}/jev_{s}.jsonl"))
    print(f"{s}: {n} rows in {OUT}/jev_{s}.jsonl", flush=True)"""),
    ("code", """# Log GPU time and pack results
hours = (time.time() - T0) / 3600
info = {"mode": MODE, "model": "openbmb/MiniCPM-OCR-Jev", "repo_commit": COMMIT, "weights_rev": WEIGHTS_REV, "dtype": DTYPE, "selfcheck": checks,
        "debug": DEBUG, "gpu_hours": round(hours, 3), "torch": torch.__version__,
        "transformers": transformers.__version__, "path": "score-only /v1/score with our HarfBuzz renders"}
json.dump(info, open(f"{OUT}/run_info.json", "w"), indent=1, default=str)
proc.terminate()
!cd /kaggle/working && rm -f jev_results.zip && zip -qr jev_results.zip jev
print(f"GPU hours this session: {hours:.2f}")"""),
]


def main():
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    for i, (kind, src) in enumerate(CELLS):
        c = {"cell_type": kind, "id": f"c{i}", "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        nb["cells"].append(c)
    out = Path(__file__).with_name("jev.ipynb")
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
