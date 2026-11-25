"""Command line: `sindhijev check crop.png "text"`, `sindhijev repair crop.png "text"`,
`sindhijev pick crop.png [--cand ...] [--tesseract]`."""
import argparse
import json
import sys


def _show(r, as_json):
    if as_json:
        return json.dumps(r, ensure_ascii=False)
    # no ^ marker line: terminals lay out right-to-left text differently, so a marker would point at the wrong letter
    return (f"p_wrong={r['p_wrong']:.3f}  {r['text']}\n"
            + (f"suspicious letters (positions {r['suspicious']}): "
               + " ".join(r["text"][i] for i in r["suspicious"]) if r["suspicious"] else "no letter flagged"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sindhijev", description="Check Sindhi OCR text against the crop image.")
    ap.add_argument("--weights", default=None, help="path to sindhijev_v2.onnx (default: SINDHIJEV_WEIGHTS / cache)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="score one OCR text for one crop")
    c.add_argument("image")
    c.add_argument("text")
    rp = sub.add_parser("repair", help="fix likely dot errors in one OCR text (keeps a swap only if it clearly wins)")
    rp.add_argument("image")
    rp.add_argument("text")
    p = sub.add_parser("pick", help="score several OCR outputs for one crop and keep the most trustworthy")
    p.add_argument("image")
    p.add_argument("--cand", action="append", default=[], help="a candidate text (repeatable)")
    p.add_argument("--tesseract", nargs="?", const="tesseract", default=None,
                   help="also run installed Tesseract models (snd, urd, ara, fas); optional path to the binary")
    a = ap.parse_args(argv)
    from .tool import Checker, tesseract_candidates  # noqa: PLC0415
    ck = Checker(a.weights)
    if a.cmd == "check":
        print(_show(ck.check(a.image, a.text), a.json))
        return 0
    if a.cmd == "repair":
        fixed, hist = ck.repair(a.image, a.text)
        print(json.dumps({"text": fixed, "changes": hist}, ensure_ascii=False) if a.json else
              "\n".join([fixed] + [f"  {h['from']} -> {h['to']}  p_wrong {h['p_wrong'][0]:.3f} -> {h['p_wrong'][1]:.3f}" for h in hist] + ([] if hist else ["  no change"])))
        return 0
    cands = {f"cand{i + 1}": t for i, t in enumerate(a.cand)}
    if a.tesseract:
        cands.update(tesseract_candidates(a.image, tess=a.tesseract))
    if not cands:
        print("give --cand TEXT and/or --tesseract", file=sys.stderr)
        return 2
    ranked = ck.pick(a.image, list(cands.values()))
    src = {}
    for k, t in cands.items():
        src.setdefault(" ".join(t.split()), []).append(k)
    for r in ranked:
        r["from"] = src.get(r["text"], [])
    if a.json:
        print(json.dumps(ranked, ensure_ascii=False))
    else:
        print("best:", _show(ranked[0], False), f"\n(from {', '.join(ranked[0]['from'])})")
        for r in ranked[1:]:
            print(f"  other p_wrong={r['p_wrong']:.3f}  {r['text']}  ({', '.join(r['from'])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
