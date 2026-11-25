# Publishing the docs site and the online playground

Nothing here runs automatically. Each step is yours to start.

## 1. Hugging Face Space (online playground + docs)

```
python scripts/export_onnx.py          # if dist/sindhijev_v2.onnx is missing
python site_src/build.py
python space/make_space.py             # -> dist/hf_space/ (~31 MB: model, site, 24 synthetic samples)
```

1. On huggingface.co: **New Space** → name `sindhijev` → SDK **Docker** → Blank → public or private.
2. Upload the contents of `dist/hf_space/` (web upload, or `git clone` the Space, copy the files in, `git lfs install`,
   commit and push; the `.onnx` and `.ttf` files go through Git LFS).
3. The Space builds the Dockerfile and serves on port 7860. Its URL looks like `https://<user>-sindhijev.hf.space`.

The Space serves the docs and the playground together, so it works on its own.

## 2. GitHub Pages (docs)

1. Repository **Settings → Pages → Build and deployment → Source: GitHub Actions**.
2. Optional: **Settings → Secrets and variables → Actions → Variables → New variable** `SINDHIJEV_API` = your Space
   URL (must start with `https://`). The Pages playground then calls the Space.
3. **Actions → Docs site → Run workflow.** The site appears at `https://<user>.github.io/sindhi-JEV/`.

Without `SINDHIJEV_API`, the Pages playground shows how to run the checker locally.

## 3. Model release and PyPI (later)

- GitHub Release: attach `dist/sindhijev_v2.onnx` and `dist/sindhijev_v2.json`, then set `WEIGHTS_URL` and
  `WEIGHTS_SHA256` in `src/sindhijev/tool.py` so `pip install`ed copies download it on first use.
- PyPI: `python -m build` then `twine upload dist/sindhijev-*` with your own PyPI token.
