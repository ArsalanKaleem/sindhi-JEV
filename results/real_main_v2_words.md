# Real-set results: main_v2 on words (1680 candidates, 336 crops)

| method | AUROC [95% CI] | AUPRC | char F1 | top-1 loc | recall@10% | recall@20% | recall@30% |
|---|---|---|---|---|---|---|---|
| model | 0.884 [0.853, 0.911] | 0.945 | 0.467 | 0.838 | 0.133 | 0.261 | 0.389 |
| main | 0.860 [0.823, 0.892] | 0.914 | 0.512 | 0.802 | 0.121 | 0.246 | 0.377 |
| abl_ref | 0.866 [0.831, 0.897] | 0.932 | 0.457 | 0.812 | 0.130 | 0.259 | 0.384 |
| abl_noscan | 0.747 [0.704, 0.789] | 0.871 | 0.337 | 0.679 | 0.125 | 0.242 | 0.360 |
| abl_norender | 0.822 [0.778, 0.860] | 0.906 | 0.485 | 0.812 | 0.126 | 0.253 | 0.374 |
| abl_noalign | 0.796 [0.747, 0.843] | 0.876 | 0.444 | 0.813 | 0.109 | 0.228 | 0.353 |
| abl_nocharhead | 0.766 [0.720, 0.811] | 0.876 | n/a | n/a | 0.124 | 0.236 | 0.355 |
| abl_v1mix | 0.823 [0.783, 0.864] | 0.900 | 0.475 | 0.817 | 0.119 | 0.242 | 0.364 |
| jev | 0.885 [0.856, 0.909] | 0.951 | n/a | n/a | 0.131 | 0.266 | 0.393 |
| ocr_conf | 0.835 [0.807, 0.860] | 0.921 | 0.025 | 0.670 | 0.130 | 0.259 | 0.389 |
| text | 0.714 [0.664, 0.761] | 0.834 | 0.437 | 0.662 | 0.108 | 0.228 | 0.340 |
| voting | 0.819 [0.788, 0.849] | 0.891 | 0.658 | 0.762 | 0.096 | 0.214 | 0.338 |

Model minus baseline, AUROC (paired, crop-clustered bootstrap):

- vs main: +0.024 [+0.002, +0.048], share of resamples where model is not better: 0.015
- vs abl_ref: +0.018 [-0.002, +0.039], share of resamples where model is not better: 0.041
- vs abl_noscan: +0.137 [+0.097, +0.174], share of resamples where model is not better: 0.000
- vs abl_norender: +0.062 [+0.036, +0.090], share of resamples where model is not better: 0.000
- vs abl_noalign: +0.088 [+0.054, +0.125], share of resamples where model is not better: 0.000
- vs abl_nocharhead: +0.118 [+0.078, +0.157], share of resamples where model is not better: 0.000
- vs abl_v1mix: +0.061 [+0.034, +0.087], share of resamples where model is not better: 0.000
- vs jev: -0.001 [-0.038, +0.037], share of resamples where model is not better: 0.527
- vs ocr_conf: +0.049 [+0.014, +0.083], share of resamples where model is not better: 0.004
- vs text: +0.170 [+0.130, +0.212], share of resamples where model is not better: 0.000
- vs voting: +0.065 [+0.030, +0.100], share of resamples where model is not better: 0.000

Engine selection (per crop, keep the output the method trusts most):

| strategy | CER | exact match |
|---|---|---|
| select_by_model | 0.215 | 0.423 |
| select_by_main | 0.231 | 0.387 |
| select_by_abl_ref | 0.232 | 0.363 |
| select_by_abl_noscan | 0.282 | 0.280 |
| select_by_abl_norender | 0.222 | 0.357 |
| select_by_abl_noalign | 0.231 | 0.363 |
| select_by_abl_nocharhead | 0.276 | 0.283 |
| select_by_abl_v1mix | 0.237 | 0.375 |
| select_by_jev | 0.234 | 0.277 |
| select_by_ocr_conf | 0.264 | 0.318 |
| select_by_text | 0.261 | 0.318 |
| select_by_voting | 0.232 | 0.304 |
| always_tess_Arabic | 0.230 | 0.286 |
| always_tess_ara | 0.381 | 0.110 |
| always_tess_fas | 0.352 | 0.119 |
| always_tess_snd | 0.181 | 0.443 |
| always_tess_urd | 0.415 | 0.077 |
| oracle | 0.131 | 0.518 |

Model ECE on real data: 0.118
