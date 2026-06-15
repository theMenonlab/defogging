# Data

Raw image datasets are not staged in this GitHub upload folder.

Recommended separate dataset uploads:

- paired fog-chamber fog images: `VerticalFilter_MediumFog_Redo_3-21-26_aligned`
- matched clear targets for those fog images: `archive_gt_matched`
- aircraft-window fog examples: one combined folder containing current and legacy examples
- free-flowing fog examples

The prepared clear-target subset contains 5,495 clear target images matching the filenames and categories in the fog-chamber fog-image folder, plus `manifest.csv` and `summary.json`.

Prepared qualitative real-fog upload folders:

- `kaggle_aircraft_window_fog_20260615`: 43 aircraft-window fog examples in one folder.
- `kaggle_free_flowing_fog_20260615`: 99 free-flowing fog examples.

External data expected by some workflows:

- Mapillary Vistas clear images for synthetic fine-tuning
- O-HAZE and NH-HAZE paired real-haze datasets
- NTIRE 2026 nighttime haze training pairs

Those third-party datasets should not be redistributed unless their licenses allow it.
