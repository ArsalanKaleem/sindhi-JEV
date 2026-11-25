# Real-set results: main_v2 on lines (370 candidates, 74 crops)

| method | AUROC [95% CI] | AUPRC | char F1 | top-1 loc | recall@10% | recall@20% | recall@30% |
|---|---|---|---|---|---|---|---|
| model | 0.458 [0.289, 0.616] | 0.951 | 0.165 | 0.708 | 0.096 | 0.198 | 0.300 |
| main | 0.656 [0.514, 0.778] | 0.975 | 0.199 | 0.688 | 0.105 | 0.207 | 0.309 |
| abl_ref | 0.697 [0.556, 0.826] | 0.980 | 0.212 | 0.708 | 0.105 | 0.210 | 0.312 |
| abl_noscan | 0.738 [0.574, 0.857] | 0.980 | 0.182 | 0.635 | 0.105 | 0.207 | 0.309 |
| abl_norender | 0.442 [0.322, 0.578] | 0.953 | 0.273 | 0.643 | 0.102 | 0.201 | 0.297 |
| abl_noalign | 0.637 [0.470, 0.777] | 0.973 | 0.186 | 0.649 | 0.105 | 0.204 | 0.306 |
| abl_nocharhead | 0.650 [0.494, 0.778] | 0.975 | n/a | n/a | 0.105 | 0.207 | 0.309 |
| abl_v1mix | 0.587 [0.360, 0.779] | 0.951 | 0.185 | 0.660 | 0.096 | 0.198 | 0.300 |
| jev | 0.910 [0.853, 0.955] | 0.995 | n/a | n/a | 0.105 | 0.210 | 0.314 |
| ocr_conf | 0.781 [0.626, 0.907] | 0.981 | 0.067 | 0.640 | 0.102 | 0.207 | 0.312 |
| text | 0.548 [0.392, 0.701] | 0.962 | 0.456 | 0.598 | 0.099 | 0.204 | 0.303 |
| voting | 0.612 [0.491, 0.749] | 0.964 | 0.646 | 0.731 | 0.093 | 0.181 | 0.286 |

Model minus baseline, AUROC (paired, crop-clustered bootstrap):

- vs main: -0.197 [-0.369, -0.061], share of resamples where model is not better: 1.000
- vs abl_ref: -0.239 [-0.371, -0.122], share of resamples where model is not better: 1.000
- vs abl_noscan: -0.280 [-0.460, -0.106], share of resamples where model is not better: 1.000
- vs abl_norender: +0.016 [-0.167, +0.181], share of resamples where model is not better: 0.436
- vs abl_noalign: -0.178 [-0.310, -0.069], share of resamples where model is not better: 1.000
- vs abl_nocharhead: -0.192 [-0.318, -0.064], share of resamples where model is not better: 1.000
- vs abl_v1mix: -0.129 [-0.253, -0.012], share of resamples where model is not better: 0.986
- vs jev: -0.451 [-0.625, -0.291], share of resamples where model is not better: 1.000
- vs ocr_conf: -0.323 [-0.529, -0.115], share of resamples where model is not better: 1.000
- vs text: -0.089 [-0.219, +0.048], share of resamples where model is not better: 0.904
- vs voting: -0.154 [-0.380, +0.036], share of resamples where model is not better: 0.924

Engine selection (per crop, keep the output the method trusts most):

| strategy | CER | exact match |
|---|---|---|
| select_by_model | 0.187 | 0.108 |
| select_by_main | 0.203 | 0.108 |
| select_by_abl_ref | 0.179 | 0.108 |
| select_by_abl_noscan | 0.231 | 0.068 |
| select_by_abl_norender | 0.211 | 0.054 |
| select_by_abl_noalign | 0.182 | 0.135 |
| select_by_abl_nocharhead | 0.237 | 0.014 |
| select_by_abl_v1mix | 0.184 | 0.041 |
| select_by_jev | 0.180 | 0.068 |
| select_by_ocr_conf | 0.186 | 0.068 |
| select_by_text | 0.213 | 0.095 |
| select_by_voting | 0.143 | 0.041 |
| always_tess_Arabic | 0.142 | 0.041 |
| always_tess_ara | 0.281 | 0.000 |
| always_tess_fas | 0.269 | 0.000 |
| always_tess_snd | 0.134 | 0.162 |
| always_tess_urd | 0.315 | 0.000 |
| oracle | 0.110 | 0.176 |

Model ECE on real data: 0.053
