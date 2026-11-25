"use strict";
(() => {
  const root = document.documentElement;
  const KEY = "sj-theme";
  const get = () => { try { return localStorage.getItem(KEY); } catch (e) { return null; } };
  const set = v => { try { localStorage.setItem(KEY, v); } catch (e) { /* private mode: theme not remembered */ } };
  const saved = get();
  if (saved) root.dataset.theme = saved;

  document.addEventListener("DOMContentLoaded", () => {
    const tb = document.getElementById("theme");
    if (tb) tb.onclick = () => {
      const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
      root.dataset.theme = dark ? "light" : "dark";
      set(root.dataset.theme);
    };
    const mb = document.getElementById("menu");
    if (mb) mb.onclick = () => document.body.classList.toggle("nav-open");

    // copy buttons on code blocks
    document.querySelectorAll("pre").forEach(pre => {
      const code = pre.querySelector("code");
      if (!code) return;
      const b = document.createElement("button");
      b.className = "copy"; b.type = "button"; b.textContent = "Copy";
      b.onclick = async () => {
        try { await navigator.clipboard.writeText(code.innerText.replace(/\n$/, "")); b.textContent = "Copied"; }
        catch (e) { b.textContent = "Select and copy"; }
        setTimeout(() => (b.textContent = "Copy"), 1400);
      };
      pre.appendChild(b);
    });

    // heading anchors + "On this page"
    const toc = document.getElementById("toc-list");
    const hs = [...document.querySelectorAll(".doc h2[id], .doc h3[id]")];
    hs.forEach(h => {
      const a = document.createElement("a");
      a.className = "anchor"; a.href = "#" + h.id; a.textContent = "#"; a.setAttribute("aria-label", "Link to this section");
      h.appendChild(a);
    });
    if (toc) {
      if (!hs.length) { toc.parentElement.style.visibility = "hidden"; return; }
      toc.innerHTML = hs.map(h => `<a href="#${h.id}" class="${h.tagName === "H3" ? "l3" : ""}">${h.firstChild.textContent}</a>`).join("");
      const links = [...toc.querySelectorAll("a")];
      const io = new IntersectionObserver(es => {
        es.forEach(e => {
          if (e.isIntersecting) links.forEach(l => l.classList.toggle("on", l.getAttribute("href") === "#" + e.target.id));
        });
      }, { rootMargin: "-70px 0px -70% 0px" });
      hs.forEach(h => io.observe(h));
    }
  });
})();
