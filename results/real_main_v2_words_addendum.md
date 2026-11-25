# Real-set results: main_v2 on words_addendum (2352 candidates, 336 crops)

| method | AUROC [95% CI] | AUPRC | char F1 | top-1 loc | recall@10% | recall@20% | recall@30% |
|---|---|---|---|---|---|---|---|
| model | 0.896 [0.867, 0.921] | 0.962 | 0.454 | 0.852 | 0.127 | 0.251 | 0.376 |
| main | 0.867 [0.831, 0.899] | 0.936 | 0.503 | 0.820 | 0.119 | 0.238 | 0.362 |
| abl_ref | 0.882 [0.851, 0.909] | 0.954 | 0.443 | 0.823 | 0.124 | 0.248 | 0.372 |
| abl_noscan | 0.767 [0.726, 0.808] | 0.903 | 0.331 | 0.695 | 0.120 | 0.236 | 0.350 |
| abl_norender | 0.846 [0.807, 0.879] | 0.936 | 0.482 | 0.832 | 0.122 | 0.244 | 0.364 |
| abl_noalign | 0.819 [0.772, 0.862] | 0.910 | 0.436 | 0.831 | 0.109 | 0.226 | 0.347 |
| abl_nocharhead | 0.782 [0.738, 0.824] | 0.905 | n/a | n/a | 0.119 | 0.231 | 0.349 |
| abl_v1mix | 0.844 [0.808, 0.879] | 0.934 | 0.459 | 0.827 | 0.122 | 0.242 | 0.358 |
| jev | 0.898 [0.868, 0.922] | 0.965 | n/a | n/a | 0.125 | 0.252 | 0.375 |
| ocr_conf | 0.821 [0.794, 0.847] | 0.930 | 0.312 | 0.608 | 0.125 | 0.246 | 0.359 |
| text | 0.699 [0.652, 0.743] | 0.865 | 0.454 | 0.674 | 0.109 | 0.226 | 0.337 |
| voting | 0.795 [0.761, 0.826] | 0.901 | 0.673 | 0.790 | 0.093 | 0.212 | 0.334 |

Model minus baseline, AUROC (paired, crop-clustered bootstrap):

- vs main: +0.028 [+0.008, +0.051], share of resamples where model is not better: 0.004
- vs abl_ref: +0.014 [-0.007, +0.032], share of resamples where model is not better: 0.093
- vs abl_noscan: +0.129 [+0.090, +0.163], share of resamples where model is not better: 0.000
- vs abl_norender: +0.050 [+0.025, +0.077], share of resamples where model is not better: 0.000
- vs abl_noalign: +0.077 [+0.045, +0.111], share of resamples where model is not better: 0.000
- vs abl_nocharhead: +0.113 [+0.077, +0.150], share of resamples where model is not better: 0.000
- vs abl_v1mix: +0.052 [+0.028, +0.075], share of resamples where model is not better: 0.000
- vs jev: -0.003 [-0.036, +0.033], share of resamples where model is not better: 0.558
- vs ocr_conf: +0.075 [+0.040, +0.108], share of resamples where model is not better: 0.000
- vs text: +0.196 [+0.158, +0.236], share of resamples where model is not better: 0.000
- vs voting: +0.101 [+0.067, +0.134], share of resamples where model is not better: 0.000

Engine selection (per crop, keep the output the method trusts most):

| strategy | CER | exact match |
|---|---|---|
| select_by_model | 0.218 | 0.423 |
| select_by_main | 0.246 | 0.405 |
| select_by_abl_ref | 0.240 | 0.372 |
| select_by_abl_noscan | 0.293 | 0.286 |
| select_by_abl_norender | 0.242 | 0.357 |
| select_by_abl_noalign | 0.255 | 0.363 |
| select_by_abl_nocharhead | 0.280 | 0.295 |
| select_by_abl_v1mix | 0.256 | 0.363 |
| select_by_jev | 0.242 | 0.280 |
| select_by_ocr_conf | 0.264 | 0.321 |
| select_by_text | 0.308 | 0.262 |
| select_by_voting | 0.391 | 0.196 |
| always_easyocr_fa | 0.442 | 0.074 |
| always_easyocr_ur | 0.468 | 0.062 |
| always_tess_Arabic | 0.230 | 0.286 |
| always_tess_ara | 0.381 | 0.110 |
| always_tess_fas | 0.352 | 0.119 |
| always_tess_snd | 0.181 | 0.443 |
| always_tess_urd | 0.415 | 0.077 |
| oracle | 0.120 | 0.542 |

Model ECE on real data: 0.108
