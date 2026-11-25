"use strict";
(() => {
  const qs = new URLSearchParams(location.search);
  const API = (qs.get("api") || window.SINDHIJEV_API || "").replace(/\/$/, "");
  const $ = id => document.getElementById(id);
  const S = { mode: "check", image: null, sample: null, samples: [] };
  const HINT = { check: "Scores how likely the text is wrong and which letters look suspicious.",
    repair: "Tries dot variants of the most suspicious letters; keeps a change only if it clearly wins (margin 0.6).",
    pick: "Scores several OCR outputs for the same crop and ranks them, most trustworthy first." };
  const LABEL = { check: "Check text", repair: "Repair dots", pick: "Compare candidates" };
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const EMPTY_DROP = "Drop an image, click to choose, or paste (Ctrl+V)<br><small>1–4 printed Sindhi words</small>";

  async function api(path, body) {
    const res = await fetch(API + path, body ? { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body) } : {});
    const o = await res.json().catch(() => ({ error: res.statusText }));
    if (!res.ok) throw new Error(o.error || res.statusText);
    return o;
  }

  function status(ok, msg) {
    const el = $("api-status");
    el.className = "call " + (ok ? "ok" : "warn");
    el.innerHTML = msg;
  }

  document.querySelectorAll(".seg button").forEach(b => b.onclick = () => {
    document.querySelectorAll(".seg button").forEach(x => x.classList.toggle("on", x === b));
    S.mode = b.dataset.mode; $("go").textContent = LABEL[S.mode]; $("modeHint").textContent = HINT[S.mode];
    $("single").hidden = S.mode === "pick"; $("multi").hidden = S.mode !== "pick";
    $("textLabel").textContent = S.mode === "pick" ? "OCR candidates" : "OCR text";
  });

  function setImage(src, label) { $("dropInner").innerHTML = `<img alt="crop" src="${src}"><br><small>${esc(label || "")}</small>`; }
  function markSample() { document.querySelectorAll(".sample").forEach(x => x.classList.toggle("on", x.dataset.key === S.sample)); }
  function readFile(f) {
    if (!f || !f.type.startsWith("image/")) return;
    const r = new FileReader();
    r.onload = () => { S.sample = null; markSample(); $("answer").hidden = true; S.image = r.result; setImage(r.result, f.name || "pasted image"); };
    r.readAsDataURL(f);
  }
  $("drop").onclick = () => $("file").click();
  $("drop").onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file").click(); } };
  $("file").onchange = e => readFile(e.target.files[0]);
  ["dragover", "dragenter"].forEach(ev => $("drop").addEventListener(ev, e => { e.preventDefault(); $("drop").classList.add("over"); }));
  ["dragleave", "drop"].forEach(ev => $("drop").addEventListener(ev, e => { e.preventDefault(); $("drop").classList.remove("over"); }));
  $("drop").addEventListener("drop", e => readFile(e.dataTransfer.files[0]));
  document.addEventListener("paste", e => {
    const it = [...(e.clipboardData || {}).items || []].find(i => i.type.startsWith("image/"));
    if (it) readFile(it.getAsFile());
  });

  const sampleUrl = k => `${API}/api/sample/${k}.png`;
  async function loadSamples() {
    S.samples = await api("/api/samples");
    $("samples").innerHTML = S.samples.map((s, i) =>
      `<div class="sample" data-key="${s.key}" title="font: ${esc(s.font)}"><img loading="lazy" alt="sample ${i + 1}" src="${sampleUrl(s.key)}"></div>`).join("");
    document.querySelectorAll(".sample").forEach(el => el.onclick = () => {
      const s = S.samples.find(x => x.key === el.dataset.key);
      S.sample = s.key; markSample(); S.image = "sample:" + s.key;
      setImage(sampleUrl(s.key), `synthetic sample · font ${s.font}`);
      $("text").value = s.cand;
      const cs = document.querySelectorAll(".cand");
      cs[0].value = s.cand; cs[1].value = s.true !== s.cand ? s.true : ""; cs[2].value = "";
      $("answer").hidden = false;
      $("answer").innerHTML = `<details><summary>Show the answer</summary>True text: <b class="ar">${esc(s.true)}</b> · the sample text ${s.cand === s.true ? "is correct" : "has an error"}</details>`;
    });
  }

  const heat = v => `rgba(214, 48, 39, ${(0.06 + 0.5 * Math.min(1, Math.max(0, v))).toFixed(3)})`;
  function chips(r, changed) {
    const flag = new Set(r.suspicious);
    return `<div class="chips">` + r.chars.map(c => c.char === " " ? `<span class="chip sp"></span>` :
      `<span class="chip${flag.has(c.i) ? " flag" : ""}${changed && changed.has(c.i) ? " changed" : ""}" style="background:${heat(c.score)}" title="letter ${c.i + 1}: ${c.score.toFixed(3)}"><div class="c">${esc(c.char)}</div><div class="s">${c.score.toFixed(2)}</div></span>`).join("") + `</div>`;
  }
  function verdict(p) {
    const bad = p >= 0.5;
    return `<div class="verdict ${bad ? "bad" : "ok"}"><div class="big">${(p * 100).toFixed(1)}%</div><div>
      <div class="what">${bad ? "Likely wrong" : "Looks correct"}</div><div class="sub">chance the text does not match the crop</div>
      <div class="meter" style="color:var(${bad ? "--bad" : "--ok"})"><i style="width:${(p * 100).toFixed(1)}%"></i></div></div></div>`;
  }
  function pair(r) {
    const scan = S.image.startsWith("sample:") ? sampleUrl(S.image.slice(7)) : S.image;
    return `<div class="pair"><figure><figcaption>Scan</figcaption><img alt="scan" src="${scan}"></figure>
      <figure><figcaption>Text as rendered for the model</figcaption><img alt="render" src="${r.render}"></figure></div>`;
  }
  const LEG = `<div class="legend"><span><i style="background:${heat(0.05)}"></i>looks fine</span><span><i style="background:${heat(0.9)}"></i>suspicious</span><span><i style="border:1.5px solid var(--bad)"></i>flagged</span></div>`;
  const view = {
    check: r => verdict(r.p_wrong) + pair(r) + `<div class="res-h">Letters, right to left</div>${chips(r)}${LEG}`,
    repair: o => {
      const ch = new Set();
      if (o.changes.length) [...o.after.text].forEach((c, i) => { if (c !== o.before.text[i]) ch.add(i); });
      return `<div class="res-h" style="margin-top:0">Before</div>${verdict(o.before.p_wrong)}${chips(o.before)}
        <div class="arrow">${o.changes.length ? "↓ " + o.changes.map(h => `${(h.p_wrong[0] * 100).toFixed(1)}% → ${(h.p_wrong[1] * 100).toFixed(1)}%`).join(", then ") : "↓ no dot change clearly wins, so the text is kept"}</div>
        <div class="res-h">After</div>${verdict(o.after.p_wrong)}${pair(o.after)}${chips(o.after, ch)}
        <div class="legend"><span><i style="border:1.5px solid var(--ok)"></i>changed letter</span></div>`;
    },
    pick: o => o.ranked.map((r, i) => `<div class="rank${i === 0 ? " best" : ""}"><div class="top"><span class="pos">${i + 1}</span>
      <span class="txt ar">${esc(r.text)}</span><span class="pill ${r.p_wrong >= 0.5 ? "bad" : "ok"}">${(r.p_wrong * 100).toFixed(1)}% wrong</span></div>${chips(r)}</div>`).join("") + LEG,
  };

  async function go() {
    const btn = $("go"), label = btn.textContent;
    btn.disabled = true; btn.innerHTML = `<span class="spin"></span>Working`;
    const out = $("out"); out.hidden = false;
    try {
      if (!S.image) throw new Error("Choose or drop a word crop first.");
      const body = { image: S.image };
      if (S.mode === "pick") {
        body.candidates = [...document.querySelectorAll(".cand")].map(x => x.value.trim()).filter(Boolean);
        if (!body.candidates.length) throw new Error("Type at least one candidate.");
      } else {
        body.text = $("text").value.trim();
        if (!body.text) throw new Error("Type the OCR text.");
      }
      const o = await api("/api/" + S.mode, body);
      out.innerHTML = view[S.mode](o) + `<p class="muted small" style="margin:12px 0 0">Model time ${o.ms} ms.</p>`;
    } catch (e) {
      out.innerHTML = `<div class="err">${esc(e.message)}</div>`;
    } finally { btn.disabled = false; btn.textContent = label; }
  }
  $("go").onclick = go;
  $("text").addEventListener("keydown", e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) go(); });
  $("reset").onclick = () => {
    S.image = null; S.sample = null; markSample(); $("text").value = "";
    document.querySelectorAll(".cand").forEach(x => (x.value = ""));
    $("dropInner").innerHTML = EMPTY_DROP; $("answer").hidden = true; $("out").hidden = true;
  };

  (async () => {
    try {
      const info = await api("/api/info");
      status(true, `<b class="t">Checker connected</b>Model v2 · letter threshold ${info.char_threshold} · temperature ${info.temperature.toFixed(3)}${API ? ` · ${esc(API)}` : " · running on this computer"}`);
      await loadSamples();
    } catch (e) {
      status(false, `<b class="t">The checker service is not reachable</b>This page needs the checker API. Run
        <code>python tools/demo/server.py</code> and open <a href="http://localhost:8770/playground.html">localhost:8770/playground.html</a>,
        or open this page with <code>?api=&lt;your Hugging Face Space URL&gt;</code>.`);
      document.querySelector(".pg").style.opacity = .55;
    }
  })();
})();
