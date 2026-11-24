"""Generate kaggle/jev.ipynb (MiniCPM-OCR Jev zero-shot on synthetic test sets, T4, <= 2 GPU h)."""
import json
from pathlib import Path

CELLS = [
    ("markdown", """# Sindhi-Jev: MiniCPM-OCR Jev zero-shot baseline

Accelerator **GPU T4 x1**, **Internet ON**, input dataset `sindhijev-jev` (from `kaggle/make_jev_bundle.py`).
First run with `DEBUG = True` (about 15 min: install, self-check, 50 items per set, speed). Then `DEBUG = False`
with **Save & Run All (commit)**, and download `jev_results.zip` from the Output tab."""),
    ("code", """# Settings
DEBUG = True
DTYPES = ["float16", "float32"]   # T4 has no bf16 (Jev's default); first dtype that passes the self-check is used
MAX_HOURS = 1.5                   # scoring time cap for this session (baseline budget: <= 2 GPU h in total)
SETS = ["test_hard", "test_font_hard"]
PORT = 18501
SELFCHECK_TOL = 0.02              # |our prob - OpenBMB reference prob| allowed on their bundled examples"""),
    ("code", """import glob, os, sys, json, time, subprocess
T0 = time.time()
hits = glob.glob("/kaggle/input/**/scripts/jev/run_jev.py", recursive=True)
assert hits, "attach the sindhijev-jev dataset"
SRC = hits[0].rsplit("/scripts/jev/", 1)[0]
OUT = "/kaggle/working/jev"
os.makedirs(OUT, exist_ok=True)
print("bundle:", SRC)
!nvidia-smi --query-gpu=name,memory.total --format=csv"""),
    ("code", """# Install Jev SDK (transformers backend) and download the weights
REPO = "/kaggle/working/MiniCPM-OCR"
if not os.path.exists(REPO):
    !git clone -q --depth 1 https://github.com/OpenBMB/MiniCPM-OCR {REPO}
COMMIT = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
JEV = f"{REPO}/scripts/minicpm-ocr-jev"
!pip install -q -r {JEV}/sdk/requirements-transformers.txt
# Kaggle's preinstalled torchaudio is built for another CUDA than the new torch, and transformers then fails to import
# AutoProcessor (seen 2026-11-24). Jev does not use audio.
!pip uninstall -y -q torchaudio
!python3 -c "from transformers import AutoProcessor, Qwen3_5ForSequenceClassification; print('import ok')"
from huggingface_hub import snapshot_download
WEIGHTS = snapshot_download("openbmb/MiniCPM-OCR-Jev")
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
                         env={**os.environ, "JEV_BACKEND_CONFIG": cfg})
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
    ("code", """# Score the synthetic test sets (resumable; per-session time cap)
for s in SETS:
    left = MAX_HOURS - (time.time() - T0) / 3600
    if left <= 0:
        break
    lim = "--limit 50" if DEBUG else ""
    !python {SRC}/scripts/jev/run_jev.py --set {SRC}/data/synth/{s} --out {OUT}/jev_{s}.jsonl --batch 16 --max-hours {left:.3f} {lim}"""),
    ("code", """# Log GPU time and pack results
hours = (time.time() - T0) / 3600
info = {"model": "openbmb/MiniCPM-OCR-Jev", "repo_commit": COMMIT, "dtype": DTYPE, "selfcheck": checks,
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
