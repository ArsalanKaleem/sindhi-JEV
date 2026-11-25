"""Build the documentation site: site_src/pages/*.html (body fragments) -> site/*.html (static, GitHub Pages ready).

    python site_src/build.py

Each page starts with a comment block:  <!-- title: ... | description: ... -->
The sidebar order below is also the prev/next order. Assets and the Sindhi font (OFL) are copied to site/assets/.
"""
import html
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "site_src"
OUT = ROOT / "site"
REPO = "https://github.com/ArsalanKaleem/sindhi-JEV"
SPACE = "https://huggingface.co/spaces/ArsalanKaleem/sindhijev"

NAV = [
    ("Get started", [("index", "Overview"), ("install", "Install"), ("quickstart", "Quick start")]),
    ("Use it", [("cli", "Command line"), ("python-api", "Python API"), ("playground", "Playground")]),
    ("Understand it", [("how-it-works", "How it works"), ("results", "Results"), ("limitations", "Limitations")]),
    ("Project", [("reproduce", "Reproduce"), ("faq", "FAQ")]),
]
ORDER = [slug for _, items in NAV for slug, _ in items]
TITLES = {slug: t for _, items in NAV for slug, t in items}

ICON_SUN = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">'
            '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2'
            'M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>')
ICON_MENU = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">'
             '<path d="M4 6h16M4 12h16M4 18h16"/></svg>')
ICON_GH = ('<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 .5a12 12 0 0 0-3.8 23.4c.6.1.8-.3.8-.6v-2c-3.3.7'
           '-4-1.6-4-1.6-.6-1.4-1.4-1.8-1.4-1.8-1.1-.7.1-.7.1-.7 1.2.1 1.9 1.3 1.9 1.3 1.1 1.8 2.8 1.3 3.5 1 .1-.8.4-1.3'
           '.8-1.6-2.7-.3-5.5-1.3-5.5-5.9 0-1.3.5-2.4 1.2-3.2-.1-.3-.5-1.5.1-3.2 0 0 1-.3 3.3 1.2a11.5 11.5 0 0 1 6 0C17.3'
           ' 4.6 18.3 5 18.3 5c.7 1.7.2 2.9.1 3.2.8.8 1.2 1.9 1.2 3.2 0 4.6-2.8 5.6-5.5 5.9.4.4.8 1.1.8 2.2v3.3c0 .3.2.7'
           '.8.6A12 12 0 0 0 12 .5z"/></svg>')

LAYOUT = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} | Sindhi-Jev</title>
<meta name="description" content="{desc}">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%231a5fd0'/%3E%3Ctext x='16' y='23' font-size='20' text-anchor='middle' fill='white' font-family='serif'%3E%D8%AC%3C/text%3E%3C/svg%3E">
<link rel="stylesheet" href="assets/style.css">
<script src="assets/site.js"></script>
</head>
<body>
<header class="top"><div class="in">
  <button class="iconbtn menu-btn" id="menu" aria-label="Menu">{menu}</button>
  <a class="brand" href="index.html"><span class="mark">ج</span>Sindhi-Jev</a>
  <nav>
    <a href="index.html"{on_docs}>Docs</a>
    <a href="playground.html"{on_play}>Playground</a>
    <a href="results.html"{on_res}>Results</a>
    <a href="{space}" target="_blank" rel="noopener">Live demo ↗</a>
  </nav>
  <span class="sp"></span>
  <a class="iconbtn" href="{repo}" aria-label="GitHub repository" title="GitHub">{gh}</a>
  <button class="iconbtn" id="theme" aria-label="Toggle light or dark theme" title="Theme">{sun}</button>
</div></header>
<div class="shell">
  <aside class="side">{side}</aside>
  <main class="content"><article class="doc">
    <div class="crumb">{section}</div>
{body}
    <nav class="pager">{pager}</nav>
    <div class="foot">Sindhi-Jev · Apache-2.0 · <a href="{repo}">source on GitHub</a> · numbers on this site come from
      the repository's result files; nothing here was tuned on the real test set.</div>
  </article></main>
  <aside class="toc"><h5>On this page</h5><div id="toc-list"></div></aside>
</div>
</body>
</html>
"""


def page_meta(text):
    m = re.match(r"\s*<!--(.*?)-->", text, re.S)
    meta = {}
    if m:
        for part in m.group(1).split("|"):
            if ":" in part:
                k, v = part.split(":", 1)
                meta[k.strip()] = v.strip()
        text = text[m.end():]
    return meta, text


def side(cur):
    out = []
    for sec, items in NAV:
        out.append(f"<h4>{sec}</h4>")
        for slug, t in items:
            out.append(f'<a href="{slug}.html"{" class=\"on\"" if slug == cur else ""}>{t}</a>')
    return "\n".join(out)


def pager(cur):
    i = ORDER.index(cur)
    parts = []
    if i > 0:
        p = ORDER[i - 1]
        parts.append(f'<a href="{p}.html"><small>Previous</small>{TITLES[p]}</a>')
    if i < len(ORDER) - 1:
        n = ORDER[i + 1]
        parts.append(f'<a class="next" href="{n}.html"><small>Next</small>{TITLES[n]}</a>')
    return "".join(parts)


def section_of(cur):
    return next(sec for sec, items in NAV if cur in [s for s, _ in items])


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "assets").mkdir(parents=True)
    for f in (SRC / "assets").iterdir():
        shutil.copy(f, OUT / "assets" / f.name)
    for f in ("NotoNaskhArabic.ttf", "OFL-NotoNaskhArabic.txt"):
        shutil.copy(ROOT / "src" / "sindhijev" / "assets" / f, OUT / "assets" / f)
    for slug in ORDER:
        meta, body = page_meta((SRC / "pages" / f"{slug}.html").read_text(encoding="utf-8"))
        on = lambda c: ' class="on"' if c else ""  # noqa: E731
        doc = LAYOUT.format(title=html.escape(meta.get("title", TITLES[slug])),
                            desc=html.escape(meta.get("description", "")), menu=ICON_MENU, sun=ICON_SUN, gh=ICON_GH,
                            repo=REPO, space=SPACE, side=side(slug), section=section_of(slug), body=body.strip(),
                            pager=pager(slug), on_docs=on(slug not in ("playground", "results")),
                            on_play=on(slug == "playground"), on_res=on(slug == "results"))
        (OUT / f"{slug}.html").write_text(doc, encoding="utf-8")
    (OUT / ".nojekyll").write_text("", encoding="utf-8")
    print(f"built {len(ORDER)} pages -> {OUT}")


if __name__ == "__main__":
    main()
