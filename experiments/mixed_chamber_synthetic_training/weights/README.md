# Model weight

Kaggle model variation:

https://www.kaggle.com/models/alingold/fog-removal/PyTorch/mixed_chamber_synthetic_nafnet

Files:

```text
nafnet_mixed_chamber_synthetic_current_ratio_20260726.pth
run_config_mixed_chamber_synthetic_current_ratio_nafnet.json
```

The checkpoint is an inference-only export using the `sigmoid_rgb` output
wrapper. Use the variation's `run_defogging_inference.py`; do not load this
checkpoint with a residual-output config.
