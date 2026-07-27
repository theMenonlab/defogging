# All-model core summary: `core_all_models`

Completed units: 270 / 270.
An arm is marked complete only when all three core datasets finished.

## Availability

| Model | Chamber | Randomized | Andrew-depth | Units |
|---|---:|---:|---:|---:|
| nafnet_fc | 3/3 | 3/3 | 3/3 | 9/9 |
| specat_realmask_l1_fc_s2 | 3/3 | 3/3 | 3/3 | 9/9 |
| reggan_fc | 3/3 | 3/3 | 3/3 | 9/9 |
| convnext_fc | 3/3 | 3/3 | 3/3 | 9/9 |
| retinexformer_fc | 3/3 | 3/3 | 3/3 | 9/9 |
| hrnet_fc | 3/3 | 3/3 | 3/3 | 9/9 |
| specat_s1_fc | 3/3 | 3/3 | 3/3 | 9/9 |
| mst_fc | 3/3 | 3/3 | 3/3 | 9/9 |
| hdnet_fc | 3/3 | 3/3 | 3/3 | 9/9 |
| padut_fc | 3/3 | 3/3 | 3/3 | 9/9 |
| dehazeformer_fog | 3/3 | 3/3 | 3/3 | 9/9 |
| swin2sr_fc | 3/3 | 3/3 | 3/3 | 9/9 |
| rdn_fc | 3/3 | 3/3 | 3/3 | 9/9 |
| restormer_fc | 3/3 | 3/3 | 3/3 | 9/9 |
| vmambair_fc | 3/3 | 3/3 | 3/3 | 9/9 |
| mirnet_fc | 3/3 | 3/3 | 3/3 | 9/9 |
| msbdn_fog | 3/3 | 3/3 | 3/3 | 9/9 |
| dcpdn_zhang_fog | 3/3 | 3/3 | 3/3 | 9/9 |
| mprnet_fc | 3/3 | 3/3 | 3/3 | 9/9 |
| sr3_fc | 3/3 | 3/3 | 3/3 | 9/9 |
| aecrnet_fog | 3/3 | 3/3 | 3/3 | 9/9 |
| ffanet_fog | 3/3 | 3/3 | 3/3 | 9/9 |
| gcanet_fog | 3/3 | 3/3 | 3/3 | 9/9 |
| pix2pix_fc | 3/3 | 3/3 | 3/3 | 9/9 |
| unetpp_fc | 3/3 | 3/3 | 3/3 | 9/9 |
| ancuti_fusion_fog | 3/3 | 3/3 | 3/3 | 9/9 |
| deanet_fog | 3/3 | 3/3 | 3/3 | 9/9 |
| griddehazenet_fog | 3/3 | 3/3 | 3/3 | 9/9 |
| dat_fc | 3/3 | 3/3 | 3/3 | 9/9 |
| drct_fc | 3/3 | 3/3 | 3/3 | 9/9 |

## PSNR winner counts

- chamber: n=30 complete model triplets; chamber=30, randomized=0, Andrew-depth=0
- synthetic_randomized: n=30 complete model triplets; chamber=0, randomized=30, Andrew-depth=0
- synthetic_depth: n=30 complete model triplets; chamber=0, randomized=0, Andrew-depth=30

## Cross-model paired PSNR summaries

Each value below is first paired over identical images within a model, then summarized over models.

| Dataset | Comparison | Models | Positive | Median delta (dB) | Range (dB) |
|---|---|---:|---:|---:|---:|
| chamber | randomized - chamber | 30 | 0 | -5.240 | -9.518 to -0.898 |
| chamber | Andrew-depth - chamber | 30 | 0 | -5.356 | -9.568 to -1.563 |
| chamber | Andrew-depth - randomized | 30 | 12 | -0.052 | -2.340 to 1.060 |
| synthetic_randomized | randomized - chamber | 30 | 30 | 6.272 | 2.695 to 10.010 |
| synthetic_randomized | Andrew-depth - chamber | 30 | 30 | 3.879 | 2.443 to 6.442 |
| synthetic_randomized | Andrew-depth - randomized | 30 | 0 | -2.078 | -4.947 to -0.106 |
| synthetic_depth | randomized - chamber | 30 | 30 | 1.887 | 0.247 to 6.094 |
| synthetic_depth | Andrew-depth - chamber | 30 | 30 | 5.127 | 2.166 to 10.388 |
| synthetic_depth | Andrew-depth - randomized | 30 | 30 | 3.408 | 0.226 to 5.992 |

## Guardrails

- Partial rows are retained for resumability but excluded from comparisons requiring missing arms.
- Positive counts refer to the named left-minus-right PSNR comparison.
- Synthetic generators have different severity distributions; compare checkpoint arms within a fixed dataset.
- Report chamber retention together with synthetic-domain gains.
- The paper-table-ready wide metric file is `all_model_metric_matrix_core_all_models.csv`.
