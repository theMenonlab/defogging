# Fog-chamber benchmark wrapper

This folder contains the wrapper used for the paired fog-chamber benchmark.

The benchmark expects:

- fog images organized by category and filename
- clear targets with the same category and filename
- a model roster supplied through a local `run6_colleague_bundle/run6` tree or equivalent third-party model source tree

The third-party model source trees are not bundled in this repository. The wrapper code is included so the direct image-to-image conversion, training loop, evaluation metrics, and result format are transparent.

The script converts older non-image-to-image model cores by rewiring the input and output boundary convolutions for direct image restoration. Native image-to-image dehazing/restoration models are loaded without that conversion.

Example validation:

```bash
python fog_rgb_benchmark.py validate \
  --fog-root data/VerticalFilter_MediumFog_Redo_3-21-26_aligned \
  --gt-root data/archive_gt_matched \
  --run6-root path/to/run6_colleague_bundle/run6
```

Example one-model training:

```bash
python fog_rgb_benchmark.py train-one \
  --model-key nafnet_fc \
  --fog-root data/VerticalFilter_MediumFog_Redo_3-21-26_aligned \
  --gt-root data/archive_gt_matched \
  --run6-root path/to/run6_colleague_bundle/run6 \
  --epochs 1 \
  --max-batches 1 \
  --max-eval-samples 2
```

The paper benchmark results are staged under `results/fog_chamber_benchmark/`.
