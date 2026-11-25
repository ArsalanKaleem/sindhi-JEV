# Jev zero-shot on synthetic sets

MiniCPM-OCR Jev as released, score-only path with our renders; run info: `{"model": "openbmb/MiniCPM-OCR-Jev", "repo_commit": "c21a3ca524263ef8166cb6d350864b10c3056b61", "dtype": "float16", "selfcheck": {"float16": {"healthz": {"status": "ok", "served_model": "/root/.cache/huggingface/hub/models--openbmb--MiniCPM-OCR-Jev/snapshots/a0ed278eb6aa1f333a711665ab610055a4a816ab", "prompt": {"template_md5": "9aab68e8ba8ab4da17ba9752bddb199c", "source": "client", "last_token": 271, "expected_last_token": 271, "token_equality": "equal", "forward_shape": "2 logits", "notes": ["the prompt we send matches build_prompt() character for character and tokenizes to 106 tokens ending on 271"]}}, "reference": {"01_text_putnam": [0.0021198309036143215, 0.00702], "02_formula_putnam": [0.9665370653480092, 0.96552], "03_table_gb_standard": [0.0009659749152204077, 0.00116]}, "pass": true}}, "debug": false, "gpu_hours": 0.386, "torch": "2.15.0+cu132", "transformers": "5.20.2", "path": "score-only /v1/score with our HarfBuzz renders"}`

| set | n | failed | Jev AUROC | v2 AUROC | v2 - Jev [95% CI] |
|---|---|---|---|---|---|
| test_hard | 2000/2000 | 0 | 0.8843 | 1.0000 | +0.1157 [+0.1014, +0.1299] |
| test_font_hard | 2000/2000 | 0 | 0.8801 | 0.9985 | +0.1184 [+0.1046, +0.1330] |
