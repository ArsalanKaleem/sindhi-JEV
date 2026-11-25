# Publishing the docs site and the online playground

Nothing here runs automatically. Each step is yours to start.

## 1. Hugging Face Space (online checker, Gradio SDK, free CPU)

```
python scripts/export_onnx.py          # if dist/sindhijev_v2.onnx is missing
python space/make_space.py             # -> dist/hf_space/ (~30 MB: app.py, model, package, 24 synthetic samples)
python dist/hf_space/app.py            # test at http://127.0.0.1:7860 before uploading
```

1. huggingface.co → **New Space** → name `sindhijev`, licence **apache-2.0**, SDK **Gradio**, template **Blank**,
   hardware **CPU Basic** (free), public → **Create Space**. (Docker is a paid option on some accounts; not needed.)
2. On the Space: **Files → Add file → Upload files**. Drag in everything *inside* `dist/hf_space/` (keep the folders
   `src/`, `model/`, `data/`), replacing the template's `README.md`. Commit.
3. The Space installs `requirements.txt` and starts `app.py` (Gradio 6.32.0, set in `README.md`). Build log:
   **Logs** tab. Address: `https://huggingface.co/spaces/<user>/sindhijev`.

The Space is a Gradio app (check / repair / compare). The docs site's playground uses this repository's own API, so
on GitHub Pages it links to the Space instead (step 2).

## 2. GitHub Pages (docs)

1. Repository **Settings → Pages → Build and deployment → Source: GitHub Actions**.
2. Optional: **Settings → Secrets and variables → Actions → Variables → New repository variable**
   `SINDHIJEV_SPACE` = your Space page, e.g. `https://huggingface.co/spaces/<user>/sindhijev`. The Pages playground then
   shows an "Open the live checker on Hugging Face" button.
3. **Actions → Docs site → Run workflow.** The site appears at `https://<user>.github.io/sindhi-JEV/`.

Without `SINDHIJEV_SPACE`, the Pages playground shows how to run the checker locally.

## 3. Model release and PyPI (later)

- GitHub Release: attach `dist/sindhijev_v2.onnx` and `dist/sindhijev_v2.json`, then set `WEIGHTS_URL` and
  `WEIGHTS_SHA256` in `src/sindhijev/tool.py` so `pip install`ed copies download it on first use.
- PyPI: `python -m build` then `twine upload dist/sindhijev-*` with your own PyPI token.
