# Code overview

This folder keeps the paper workflows separated by role.

- `nafnet_finetuning/`: reusable NAFNet training, evaluation, and inference utilities.
- `synthetic_finetuning/`: spatial synthetic-fog generator and Mapillary synthetic fine-tuning workflow.
- `fog_chamber_benchmark/`: benchmark wrapper for paired fog-chamber restoration experiments. Third-party model repositories are not bundled.
- `classification_semantic_preservation/`: clear-trained ResNet-50 semantic-preservation check.
- `dark_channel_prior/`: classical dark-channel-prior comparison.
- `fog_statistics/`: fog proxy, PSD, and paired structure-loss analyses.

Most scripts accept command-line paths. The release package avoids hard-coded private dataset paths where practical, but external datasets and checkpoints still need to be supplied locally before running training or evaluation.
