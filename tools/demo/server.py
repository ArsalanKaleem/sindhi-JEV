"""Local web demo of the sindhijev tool (no publishing, runs on this machine only).

    python tools/demo/server.py [--port 8770] [--weights dist/sindhijev_v2.onnx]
    open http://localhost:8770

API (JSON): POST /api/check {image, text} | /api/repair {image, text} | /api/pick {image, candidates[]}
`image` is a data URL / base64 PNG/JPEG, or "sample:<set>/<id>" for a synthetic sample. GET /api/samples lists
synthetic samples (never the real test set). Responses add the candidate render as a PNG data URL.
"""
import argparse
import base64
import json
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sindhijev.tool import Checker  # noqa: E402

HERE = Path(__file__).resolve().parent
SAMPLE_SETS = ["test_font_hard", "test_font"]  # synthetic, unseen fonts; real crops are never served
CK = None
SAMPLES = []
LOCK = threading.Lock()  # one model call at a time (renderer state is shared)


def load_samples(n_per_set=12, seed=0):
    rng = random.Random(seed)
    out = []
    for s in SAMPLE_SETS:
        rows = [json.loads(x) for x in open(ROOT / "data" / "synth" / s / "manifest.jsonl", encoding="utf-8")]
        pos = [r for r in rows if r["mode"] == "positive"]
        neg = [r for r in rows if r["mode"] != "positive"]
        for r in rng.sample(pos, n_per_set // 2) + rng.sample(neg, n_per_set // 2):
            out.append({"key": f"{s}/{r['id']}", "set": s, "cand": r["cand"], "true": r["true"], "mode": r["mode"],
                        "font": r["font"].rsplit(".", 1)[0]})
    rng.shuffle(out)
    return out


def png_url(img):
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def decode_image(spec):
    if spec.startswith("sample:"):
        key = spec[len("sample:"):]
        if key not in {s["key"] for s in SAMPLES}:
            raise ValueError("unknown sample")
        s, i = key.split("/")
        img = cv2.imread(str(ROOT / "data" / "synth" / s / "scan" / f"{i}.png"), cv2.IMREAD_GRAYSCALE)
    else:
        raw = base64.b64decode(spec.split(",", 1)[-1])
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("could not read the image")
    if img.shape[0] > 400:  # word crops: keep it light
        img = cv2.resize(img, (int(img.shape[1] * 400 / img.shape[0]), 400), interpolation=cv2.INTER_AREA)
    return img


def with_render(r):
    img, _ = CK.r.render(r["text"] or " ")
    return {**r, "render": png_url(img)}


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/samples":
            return self._send(200, SAMPLES)
        if self.path.startswith("/api/sample/"):
            key = self.path[len("/api/sample/"):].removesuffix(".png")
            if key not in {s["key"] for s in SAMPLES}:
                return self._send(404, {"error": "unknown sample"})
            s, i = key.split("/")
            return self._send(200, (ROOT / "data" / "synth" / s / "scan" / f"{i}.png").read_bytes(), "image/png")
        if self.path == "/font.ttf":
            return self._send(200, (ROOT / "src" / "sindhijev" / "assets" / "NotoNaskhArabic.ttf").read_bytes(), "font/ttf")
        if self.path == "/api/info":
            return self._send(200, {"temperature": CK.temperature, "char_threshold": CK.char_thr,
                                    "repair_margin": 0.6})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > 8_000_000:
                return self._send(413, {"error": "image too large (max ~6 MB)"})
            req = json.loads(self.rfile.read(n) or b"{}")
            img = decode_image(req.get("image", ""))
        except Exception as e:  # noqa: BLE001
            return self._send(400, {"error": str(e)})
        with LOCK:
            return self._compute(req, img)

    def _compute(self, req, img):
        try:
            t0 = time.time()
            if self.path == "/api/check":
                out = with_render(CK.check(img, req.get("text", "")))
            elif self.path == "/api/repair":
                fixed, hist = CK.repair(img, req.get("text", ""))
                out = {"before": with_render(CK.check(img, req.get("text", ""))),
                       "after": with_render(CK.check(img, fixed)), "changes": hist}
            elif self.path == "/api/pick":
                cands = [c for c in req.get("candidates", []) if c and c.strip()][:8]
                if not cands:
                    return self._send(400, {"error": "give at least one candidate"})
                out = {"ranked": [with_render(r) for r in CK.pick(img, cands)]}
            else:
                return self._send(404, {"error": "not found"})
            out["ms"] = round((time.time() - t0) * 1000)
            return self._send(200, out)
        except Exception as e:  # noqa: BLE001 (demo: report the problem to the page)
            return self._send(400, {"error": str(e)})

    def log_message(self, fmt, *args):
        sys.stderr.write("%s\n" % (fmt % args))


def main():
    global CK, SAMPLES
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--weights", default=str(ROOT / "dist" / "sindhijev_v2.onnx"))
    a = ap.parse_args()
    CK = Checker(a.weights)
    SAMPLES = load_samples()
    print(f"sindhijev demo on http://localhost:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
