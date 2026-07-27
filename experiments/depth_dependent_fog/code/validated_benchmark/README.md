# Validated depth-dependent training workflow

This is the code path used for the completed NAFNet checkpoint and matched
30-model experiment. `synthetic_finetune_all_models.py` supports both
`randomized` and `andrew_depth`; use `andrew_depth` for this extension.

## Inputs

- Mapillary Vistas with `training/images`, `validation/images`, and
  `testing/images`.
- A mirrored ZoeDepth cache.
- The fog-chamber checkpoint root containing
  `nafnet_fc/final.pth`.
- The included benchmark registry and minimal NAFNet source.

## 1. Install

From the experiment root:

```bash
pip install -r requirements.txt
```

## 2. Precompute ZoeDepth

```bash
python code/validated_benchmark/precompute_depth.py \
  --input /path/to/mapillary_vistas \
  --output /path/to/mapillary_vistas_depth_zoe \
  --workers 4
```

The operation is recursive, resumable, and atomic. CUDA preprocessing uses
float16, matching the completed run. The cache mirrors all relative paths and
stores `.npy` files.

## 3. Smoke test

```bash
python code/validated_benchmark/synthetic_finetune_all_models.py \
  --model-key nafnet_fc \
  --fog-mode andrew_depth \
  --mapillary-root /path/to/mapillary_vistas \
  --depth-root /path/to/mapillary_vistas_depth_zoe \
  --preset-json code/validated_benchmark/presets/andrew_depth.json \
  --benchmark-script code/validated_benchmark/benchmark/fog_rgb_benchmark.py \
  --run6-root code/validated_benchmark/benchmark/run6 \
  --init-checkpoint-root /path/to/chamber_checkpoints \
  --output-root /path/to/output \
  --crop-size 512 \
  --smoke
```

## 4. Reproduce the released NAFNet

Run the same command without `--smoke` and add:

```text
--micro-batch 1
--accumulation-steps 2
--epochs 1
--max-optimizer-steps 2600
--max-val-batches 300
--max-test-batches 500
--learning-rate 0.0001
--weight-decay 0.0001
--color-loss-weight 0.02
--residual-tv-weight 0.008
--seed 734
```

The full 30-model experiment additionally requires the adapted third-party
model source trees used by the repository benchmark roster. Those upstream
sources are not duplicated here.
