# Model weight

Kaggle dataset:

https://www.kaggle.com/datasets/themenonlab/computational-defogging-model-weights

Files:

```text
nafnet_mixed_chamber_synthetic_current_ratio_20260726.pth
run_config_mixed_chamber_synthetic_current_ratio_nafnet.json
```

The checkpoint is an inference-only export using the `sigmoid_rgb` output
wrapper. Use the Kaggle dataset's `run_defogging_inference.py`; do not load this
checkpoint with a residual-output config.
