# Staging notes

Prepared on 2026-06-15.

This staging folder is intended to become the GitHub repository contents. It excludes:

- raw image datasets
- Mapillary crops
- O-HAZE/NH-HAZE/NTIRE image data
- trained checkpoint binaries
- rendered prediction folders
- demo videos
- paper LaTeX snapshots
- generated SLURM scripts and private CHPC launch files
- Python caches and backup files

Included lightweight results:

- fog-chamber benchmark per-model `metrics.csv` and `summary.json`
- fog-chamber benchmark aggregate summary table
- current NAFNet run summaries, split manifests, and training histories
- supplement table CSV/JSON sources
- figure metric CSVs used for manuscript tables

The fog-chamber benchmark wrapper is included, but third-party model source trees are not bundled. Users who want to rerun the full benchmark need to supply the corresponding third-party model implementations and confirm their licenses.
