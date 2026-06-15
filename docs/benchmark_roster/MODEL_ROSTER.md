# Canonical Model Roster

This bundle keeps one benchmark train/test pair per benchmarked model key.

## Legacy models

| Model key | Directory | Train script | Test script |
|---|---|---|---|
| `convnext_fc` | `run6/phamscope_convnext` | `train_convnext_fc.py` | `test_convnext_fc.py` |
| `dat_fc` | `run6/phamscope_dat` | `train_dat_fc.py` | `test_dat_fc.py` |
| `drct_fc` | `run6/phamscope_drct` | `train_drct_fc.py` | `test_drct_fc.py` |
| `edsr_fc` | `run6/phamscope_edsr` | `train_edsr_fc.py` | `test_edsr_fc.py` |
| `gmsr_fc` | `run6/phamscope_gmsr` | `train_gmsr_fc.py` | `test_gmsr_fc.py` |
| `hat_fc` | `run6/phamscope_hat` | `train_hat_fc.py` | `test_hat_fc.py` |
| `hdnet_fc` | `run6/phamscope_hdnet` | `train_hdnet_fc.py` | `test_hdnet_fc.py` |
| `hrnet_fc` | `run6/phamscope_hrnet` | `train_hrnet_fc.py` | `test_hrnet_fc.py` |
| `hscnn_fc` | `run6/phamscope_hscnn` | `train_hscnn_fc.py` | `test_hscnn_fc.py` |
| `hsrmamba_fc` | `run6/phamscope_hsrmamba` | `train_hsrmamba_fc.py` | `test_hsrmamba_fc.py` |
| `mirnet_fc` | `run6/phamscope_mirnet` | `train_mirnet_fc.py` | `test_mirnet_fc.py` |
| `mprnet_fc` | `run6/phamscope_mprnet` | `train_mprnet_fc.py` | `test_mprnet_fc.py` |
| `mst_fc` | `run6/phamscope_mst` | `train_mst_fc.py` | `test_mst_fc.py` |
| `nafnet_fc` | `run6/phamscope_nafnet` | `train_nafnet_fc.py` | `test_nafnet_fc.py` |
| `padut_fc` | `run6/phamscope_padut` | `train_padut_fc.py` | `test_padut_fc.py` |
| `pix2pix_fc` | `run6/phamscope_pix2pix` | `train_pix2pix_fc.py` | `test_pix2pix_fc.py` |
| `rdn_fc` | `run6/phamscope_rdn` | `train_rdn_fc.py` | `test_rdn_fc.py` |
| `reggan_fc` | `run6/phamscope_reggan` | `train_reggan_fc.py` | `test_reggan_fc.py` |
| `restormer_fc` | `run6/phamscope_restormer` | `train_restormer_fc.py` | `test_restormer_fc.py` |
| `retinexformer_fc` | `run6/phamscope_retinexformer` | `train_retinexformer_fc.py` | `test_retinexformer_fc.py` |
| `sr3_fc` | `run6/phamscope_sr3` | `train_sr3_fc.py` | `test_sr3_fc.py` |
| `swin2sr_fc` | `run6/phamscope_swin2sr` | `train_swin2sr_fc.py` | `test_swin2sr_fc.py` |
| `swinir_fc` | `run6/phamscope_swinir` | `train_swinir_fc.py` | `test_swinir_fc.py` |
| `unetpp_fc` | `run6/phamscope_unetpp` | `train_unetpp_fc.py` | `test_unetpp_fc.py` |
| `vmambair_fc` | `run6/phamscope_vmambair` | `train_vmambair_fc.py` | `test_vmambair_fc.py` |

## SpecAT variants

| Model key | Directory | Train script | Test script |
|---|---|---|---|
| `specat_s1_fc` | `run6/phamscope_specat` | `train_specat_s1_fc.py` | `test_specat_s1_fc.py` |
| `specat_s2_fc` | `run6/phamscope_specat` | `train_specat_s2_fc.py` | `test_specat_s2_fc.py` |
| `specat_realmask_l1_fc_s1` | `run6/phamscope_specat` | `train_specat_realmask_l1_fc_s1.py` | `test_specat_realmask_l1_fc_s1.py` |
| `specat_realmask_l1_fc_s2` | `run6/phamscope_specat` | `train_specat_realmask_l1_fc_s2.py` | `test_specat_realmask_l1_fc_s2.py` |
| `specat_realmask_l1_s1_synth_optcal_10e_ft_fc` | `run6/phamscope_specat` | `train_specat_realmask_l1_s1_synth_optcal_ft_fc.py` | `test_specat_realmask_l1_s1_synth_optcal_ft_fc.py` |
| `specat_realmask_l1_s2_synth_optcal_10e_ft_fc` | `run6/phamscope_specat` | `train_specat_realmask_l1_s2_synth_optcal_ft_fc.py` | `test_specat_realmask_l1_s2_synth_optcal_ft_fc.py` |

## SCOPE variants

| Model key | Directory | Train script | Test script |
|---|---|---|---|
| `scope_fc` | `run6/phamscope_bassai` | `train_scope_fc.py` | `test_scope_fc.py` |
| `scope_fc_ft20` | `run6/phamscope_bassai` | `train_scope_fc.py` | `test_scope_fc.py` |
| `scope_fc_ft20_nocropshift` | `run6/phamscope_bassai` | `train_scope_fc.py` | `test_scope_fc.py` |

## Latest big-dataset research entrypoints

These are the preferred latest research pipelines for the big multi-holdout benchmark:

- SCOPE: `run6/big_redo_scope_final.py`
- SpecAT: `run6/big_redo_extra_specat.py`
- Legacy roster: `run6/big_redo_legacy.py`

## Recommended usage

Most users should use the top-level wrappers instead of calling internal scripts directly:

```bash
bash run_one_model.sh scope
bash run_one_model.sh specat
bash run_one_model.sh legacy convnext_fc
bash run_full_run6.sh
```

## Native RGB fog-removal additions

These entries are used by `20260523_fog_benchmarking/fog_rgb_benchmark.py`.
They expose the same `build_model()` convention as the legacy train scripts,
but are native `3 -> 3` RGB dehazing models rather than `1 -> 120` PHAMscope
models.

| Model key | Directory | Train/build script |
| --- | --- | --- |
| `dehazeformer_fog` | `run6/phamscope_dehazeformer` | `train_dehazeformer_fog.py` |
| `dcpdn_zhang_fog` | `run6/phamscope_dcpdn_zhang` | `train_dcpdn_zhang_fog.py` |
| `ancuti_fusion_fog` | `run6/phamscope_ancuti_fusion` | `train_ancuti_fusion_fog.py` |
| `ffanet_fog` | `run6/phamscope_ffanet` | `train_ffanet_fog.py` |
| `griddehazenet_fog` | `run6/phamscope_griddehazenet` | `train_griddehazenet_fog.py` |
| `aodnet_fog` | `run6/phamscope_aodnet` | `train_aodnet_fog.py` |
| `gcanet_fog` | `run6/phamscope_gcanet` | `train_gcanet_fog.py` |
| `msbdn_fog` | `run6/phamscope_msbdn` | `train_msbdn_fog.py` |
| `aecrnet_fog` | `run6/phamscope_aecrnet` | `train_aecrnet_fog.py` |
| `deanet_fog` | `run6/phamscope_deanet` | `train_deanet_fog.py` |
| `nafnet_bottleneck1_fog` | `run6/phamscope_nafnet_bottleneck1` | `train_nafnet_bottleneck1_fog.py` |
| `nafnet_no_sca_fog` | `run6/phamscope_nafnet_no_sca` | `train_nafnet_no_sca_fog.py` |
