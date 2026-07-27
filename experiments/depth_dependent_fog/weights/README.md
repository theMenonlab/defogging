# Model weight

Kaggle dataset:

https://www.kaggle.com/datasets/themenonlab/computational-defogging-model-weights

Files:

```text
nafnet_depth_dependent_synthetic_20260722.pth
run_config_depth_dependent_synthetic_nafnet.json
```

The checkpoint is an inference-only export using the `residual_rgb` output
wrapper. Use the Kaggle dataset's `run_defogging_inference.py`; do not load this
checkpoint with the mixed model's `sigmoid_rgb` config.
