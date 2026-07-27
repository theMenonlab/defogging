# Synthetic Fine-Tuning Workflow

This folder contains the paper-current synthetic fine-tuning code.

The released synthetic fine-tuned NAFNet starts from the fog-chamber NAFNet checkpoint and trains on clear Mapillary Vistas crops with spatial synthetic fog generated on the fly.

Core files:

- `spatial_fog_model.py`: spatial synthetic-fog generator.
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
