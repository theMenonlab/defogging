# Synthetic fine-tuning workflow

This is the paper-current synthetic fine-tuning code.

Core files:

- `spatial_fog_model.py`: spatial synthetic-fog generator.
- `train_spatial_mapillary_nafnet.py`: trains NAFNet on clear Mapillary crops with synthetic fog generated on the fly.
- `run_followup_synthetic_fog_experiments.py`: runner containing the final synthetic fine-tuning branch used for the paper.
- `run_no_pretraining_ablation.py`: synthetic fine-tuning ablation without fog-chamber initialization.
- `evaluate_public_paired_checkpoint.py`: direct-transfer evaluation on public paired haze examples.
- `summarize_public_eval_extended.py`: summary table builder for public paired evaluation outputs.
- `make_experiment_review_sheets.py`: review-sheet helper used during model selection.

The final paper should call this branch the `synthetic fine-tuned NAFNet`; avoid internal run labels in public prose.

Required external inputs:

- fog-chamber NAFNet checkpoint
- Mapillary Vistas clear images
- optional public paired-haze datasets for direct-transfer checks
- optional aircraft-window and free-flowing fog examples for qualitative inference

The scripts still contain local workstation paths. Keep them as provenance in this staging copy, or convert them to environment-variable defaults before public upload.

