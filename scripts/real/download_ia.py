"""Download the approved archive.org PDFs (docs/real_sources_shortlist.md) into data/real/raw_pdfs/.

Reads the exact original file name from the saved item metadata, downloads with resume,
verifies size (and md5 when archive.org lists it). Never executes anything from the files.
    python scripts/real/download_ia.py --meta <dir with <id>.json> --out data/real/raw_pdfs
"""
import argparse, hashlib, json, sys, time, urllib.parse, urllib.request
from pathlib import Path

ITEMS = {  # source_id -> (archive.org id, exact PDF name prefix to pick)
    "ia_trumpp1872": "grammarsindhila03trumgoog", "ia_bcp1873": "sindi_bcp1873", "ia_tanbih1872": "san_0460",
    "ia_chachnamo1923": "1923_20260819_202608", "ia_sindhu1938": "san_0046", "ia_kalimat1934": "san_0103",
    "ia_raddmizan1936": "RaddMizan-ul-insafsindhi1936", "ia_mehran1956": "5-1-1956", "ia_mehran1959": "san_0850",
    "ia_lughat1960": "san_0836", "ia_baloch1961": "soorih-sindhi-adab_20250312_1710", "ia_dharejo2002": "2002_20231018",
    "ia_akhlaqiat2020": "1512272497_202106", "ia_dildegree2025": "dil-degree-duty-.pdf-1",
}

def pick_pdf(meta):
    pdfs = [f for f in meta["files"] if f["name"].lower().endswith(".pdf")]
    orig = [f for f in pdfs if f.get("source") == "original"] or pdfs
    return max(orig, key=lambda f: int(f.get("size", 0) or 0))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--meta", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = []
    for sid, ia in ITEMS.items():
        meta = json.load(open(Path(a.meta) / f"{ia}.json", encoding="utf-8"))
        f = pick_pdf(meta); size = int(f.get("size", 0) or 0)
        dst = out / f"{sid}.pdf"
        url = f"https://archive.org/download/{ia}/{urllib.parse.quote(f['name'])}"
        if not (dst.exists() and dst.stat().st_size == size):
            for attempt in range(3):
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "research-downloader/0.1"})
                    with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as w:
                        while chunk := r.read(1 << 20):
                            w.write(chunk)
                    break
                except Exception as e:
                    print(sid, "retry", attempt, e, flush=True); time.sleep(5)
        got = dst.stat().st_size if dst.exists() else 0
        md5 = hashlib.md5(dst.read_bytes()).hexdigest() if got else ""
        ok = got == size and (not f.get("md5") or f["md5"] == md5)
        log.append({"source_id": sid, "ia_id": ia, "file": f["name"], "url": url, "bytes": got, "expected": size, "md5": md5, "ok": ok})
        print(f"{'OK ' if ok else 'BAD'} {sid} {got/1e6:.1f} MB", flush=True)
    (out / "download_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    sys.exit(0 if all(x["ok"] for x in log) else 1)

if __name__ == "__main__":
    main()
