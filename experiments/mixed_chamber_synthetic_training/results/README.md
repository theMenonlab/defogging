# Result files

Each ratio directory contains its sanitized `run_config.json` and
`summary.json`. The checkpoints are intentionally excluded from Git.

- `current_ratio/`: 247,150 chamber + 2,600 synthetic updates; released model.
- `synthetic_0p2x/`: 247,150 chamber + 520 synthetic updates.
- `synthetic_5x/`: 247,150 chamber + 13,000 synthetic updates.
- `expanded_comparison_metrics.csv`: image-level metrics for the expanded
  qualitative comparison.
- `expanded_comparison_results.md`: provenance and interpretation notes.

The 0.2x job completed after the July 26 paper package was frozen, so its result
is documented here but is not retroactively added to that paper PDF.
