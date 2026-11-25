# Real-set results: main_v2 on lines_addendum (518 candidates, 74 crops)

| method | AUROC [95% CI] | AUPRC | char F1 | top-1 loc | recall@10% | recall@20% | recall@30% |
|---|---|---|---|---|---|---|---|
| model | 0.486 [0.313, 0.649] | 0.966 | 0.167 | 0.737 | 0.098 | 0.200 | 0.299 |
| main | 0.682 [0.541, 0.803] | 0.984 | 0.201 | 0.717 | 0.104 | 0.206 | 0.305 |
| abl_ref | 0.708 [0.571, 0.837] | 0.986 | 0.207 | 0.735 | 0.104 | 0.208 | 0.307 |
| abl_noscan | 0.754 [0.582, 0.874] | 0.987 | 0.188 | 0.667 | 0.104 | 0.206 | 0.305 |
| abl_norender | 0.433 [0.310, 0.568] | 0.965 | 0.268 | 0.693 | 0.102 | 0.202 | 0.297 |
| abl_noalign | 0.642 [0.472, 0.784] | 0.981 | 0.179 | 0.659 | 0.104 | 0.204 | 0.303 |
| abl_nocharhead | 0.669 [0.509, 0.801] | 0.983 | n/a | n/a | 0.104 | 0.206 | 0.305 |
| abl_v1mix | 0.600 [0.370, 0.796] | 0.964 | 0.180 | 0.705 | 0.098 | 0.200 | 0.299 |
| jev | 0.918 [0.863, 0.961] | 0.997 | n/a | n/a | 0.104 | 0.208 | 0.309 |
| ocr_conf | 0.836 [0.710, 0.934] | 0.992 | 0.300 | 0.567 | 0.104 | 0.206 | 0.307 |
| text | 0.558 [0.401, 0.717] | 0.973 | 0.463 | 0.643 | 0.100 | 0.204 | 0.303 |
| voting | 0.592 [0.469, 0.728] | 0.973 | 0.661 | 0.794 | 0.084 | 0.188 | 0.289 |

Model minus baseline, AUROC (paired, crop-clustered bootstrap):

- vs main: -0.196 [-0.379, -0.053], share of resamples where model is not better: 1.000
- vs abl_ref: -0.222 [-0.355, -0.101], share of resamples where model is not better: 1.000
- vs abl_noscan: -0.268 [-0.456, -0.085], share of resamples where model is not better: 1.000
- vs abl_norender: +0.054 [-0.129, +0.217], share of resamples where model is not better: 0.273
- vs abl_noalign: -0.156 [-0.282, -0.053], share of resamples where model is not better: 0.999
- vs abl_nocharhead: -0.182 [-0.310, -0.046], share of resamples where model is not better: 0.998
- vs abl_v1mix: -0.114 [-0.240, -0.001], share of resamples where model is not better: 0.975
- vs jev: -0.431 [-0.611, -0.268], share of resamples where model is not better: 1.000
- vs ocr_conf: -0.350 [-0.559, -0.151], share of resamples where model is not better: 1.000
- vs text: -0.071 [-0.210, +0.072], share of resamples where model is not better: 0.830
- vs voting: -0.105 [-0.345, +0.095], share of resamples where model is not better: 0.797

Engine selection (per crop, keep the output the method trusts most):

| strategy | CER | exact match |
|---|---|---|
| select_by_model | 0.203 | 0.095 |
| select_by_main | 0.220 | 0.108 |
| select_by_abl_ref | 0.200 | 0.108 |
| select_by_abl_noscan | 0.248 | 0.054 |
| select_by_abl_norender | 0.230 | 0.027 |
| select_by_abl_noalign | 0.193 | 0.122 |
| select_by_abl_nocharhead | 0.246 | 0.014 |
| select_by_abl_v1mix | 0.201 | 0.041 |
| select_by_jev | 0.187 | 0.068 |
| select_by_ocr_conf | 0.186 | 0.068 |
| select_by_text | 0.230 | 0.068 |
| select_by_voting | 0.321 | 0.014 |
| always_easyocr_fa | 0.323 | 0.000 |
| always_easyocr_ur | 0.345 | 0.000 |
| always_tess_Arabic | 0.142 | 0.041 |
| always_tess_ara | 0.281 | 0.000 |
| always_tess_fas | 0.269 | 0.000 |
| always_tess_snd | 0.134 | 0.162 |
| always_tess_urd | 0.315 | 0.000 |
| oracle | 0.110 | 0.176 |

Model ECE on real data: 0.039
