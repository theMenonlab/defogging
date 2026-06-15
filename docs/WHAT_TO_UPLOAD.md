# Upload checklist

## GitHub repository

Commit the contents of this staging folder as ordinary Git files.

## GitHub Release assets

Attach these checkpoint files as release assets rather than committing them:

- `fog_chamber_nafnet_model_state_20260615.pth`
- `synthetic_finetuned_nafnet_model_state_20260615.pt`
- `SHA256SUMS.txt`

Do not upload task-specific O-HAZE/NH-HAZE or NTIRE checkpoints for this release.

## Kaggle dataset upload

Upload the fog-chamber fog images and matched clear targets together:

- fog images: `VerticalFilter_MediumFog_Redo_3-21-26_aligned`
- matched clear targets: `archive_gt_matched`
- metadata: `manifest.csv`
- summary: `summary.json`

Upload qualitative real-fog examples:

- aircraft-window fog examples: one combined folder containing current and legacy examples
- free-flowing fog examples

## Do not upload to Git

- checkpoint binaries
- raw datasets
- rendered prediction folders
- demo videos
- private CHPC scripts
- local backups
