# Synthetic Fine-Tuning Workflow

This folder contains the paper-current synthetic fine-tuning code.

The released synthetic fine-tuned NAFNet starts from the fog-chamber NAFNet checkpoint and trains on clear Mapillary Vistas crops with spatial synthetic fog generated on the fly.

Core files:

- `spatial_fog_model.py`: spatial synthetic-fog generator.
- `precompute_depth.py`: resumable ZoeDepth cache generator.
- `train_spatial_mapillary_nafnet.py`: trains NAFNet on Mapillary crops with synthetic fog.
- `run_followup_synthetic_fog_experiments.py`: provenance runner for the final paper branch.
- `run_no_pretraining_ablation.py`: synthetic fine-tuning ablation without fog-chamber initialization.
- `evaluate_public_paired_checkpoint.py`: direct-transfer evaluation on paired public haze examples.
- `summarize_public_eval_extended.py`: summary table builder for paired evaluation outputs.

Required inputs:

- fog-chamber NAFNet checkpoint from https://www.kaggle.com/models/alingold/fog-removal
- Mapillary Vistas clear images from https://www.kaggle.com/datasets/kaggleprollc/mapillary-vistas-image-data-collection
- optional public paired-haze datasets for transfer checks
- optional aircraft-window and free-flowing fog examples for qualitative inference

This is an advanced GPU workflow. For ordinary use, run the released synthetic fine-tuned checkpoint with `code/nafnet_finetuning/run_defogging_inference.py`.

## ZoeDepth cache

Use Transformers 4.46 or newer. Precompute the complete Mapillary tree once:

```bash
python code/synthetic_finetuning/precompute_depth.py \
  --input "/path/to/Mapillary Vistas" \
  --output /path/to/mapillary_depth \
  --workers 4
```

The output mirrors the input, for example
`mapillary_depth/training/images/example.npy`. The command is resumable. For a
one-image GPU smoke test, add `--max-images 1`.

CUDA preprocessing defaults to float16 because that is the precision used for
the completed depth-based run. Invalid depth pixels are replaced with the
finite median before saving. `--precision float32` is available for new
experiments, but it changes the metric-depth scale and therefore requires fog
parameter retuning.

Pass the cache root to training:

```bash
python code/synthetic_finetuning/train_spatial_mapillary_nafnet.py \
  --mapillary-root "/path/to/Mapillary Vistas" \
  --depth-root /path/to/mapillary_depth \
  --preset-json code/synthetic_finetuning/spatial_fog_preset.json \
  --out-dir outputs \
  --init-checkpoint /path/to/fog_chamber_checkpoint.pth \
  --strict-load
```
