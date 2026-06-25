# Computational defogging code release

This repository contains the code and lightweight result files used for the computational defogging paper.

Large image datasets, trained checkpoints, rendered prediction folders, and videos are released on kaggle: https://www.kaggle.com/models/alingold/fog-removal
https://www.kaggle.com/datasets/alingold/fog-chamber

## Repository layout

- `code/nafnet_finetuning/`: NAFNet training, inference, checkpoint, and evaluation utilities.
- `code/synthetic_finetuning/`: spatial synthetic-fog generation and synthetic fine-tuning workflow.
- `code/fog_chamber_benchmark/`: fog-chamber benchmark wrapper used to train and evaluate restoration models against paired fog/clear images.
- `code/classification_semantic_preservation/`: semantic-preservation classification checks.
- `code/dark_channel_prior/`: dark-channel-prior baseline.
- `code/fog_statistics/`: fog statistics and paired-image structure analyses.
- `results/`: small CSV/JSON result tables and split manifests.
- `models/`: checkpoint manifest and release notes.
- `data/`: expected dataset layout and Kaggle upload notes.
- `docs/`: staging notes and upload checklist.

## Model names

Use these paper-facing names in public documentation:

- fog-chamber NAFNet
- synthetic fine-tuned NAFNet
- mixed O-HAZE/NH-HAZE task-specific NAFNet
- NTIRE task-specific NAFNet

## Checkpoints

The trained NAFNet checkpoints are not stored in Git because each model-state file is about 112 MB. See `models/checkpoints_manifest.csv` for expected release-asset filenames and SHA256 hashes.

## License

Code in this staging package is prepared for release under the MIT License. Third-party datasets and third-party model implementations have their own licenses and are not bundled here.
