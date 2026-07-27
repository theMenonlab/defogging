# Depth-dependent synthetic fog extension

Status: experimental post-submission extension. This method and checkpoint are
not part of the final PNAS submission.

This directory preserves the ZoeDepth-dependent fog work that previously
replaced the repository's default randomized generator. The repository root has
been restored to the submission workflow; this extension now has its own code,
results, paper variant, and checkpoint link.

## Method

ZoeDepth (`Intel/zoedepth-nyu-kitti`) is precomputed for every Mapillary image.
During fog rendering, raw positive depth modulates the spatial attenuation:

```text
beta(x) = clip(beta_mean * (1 + paint_weight * zoe_depth(x)), 0.03, 8)
T(x) = exp(-beta(x) * base_depth(x))
```

The released NAFNet starts from the 50-epoch fog-chamber checkpoint and uses
one epoch capped at 2,600 optimizer updates, effective batch size 2, 512-pixel
crops, AdamW at `1e-4`, and seed 734.

## Evidence

Matched one-epoch experiments were completed for 30 architectures.

- Every randomized checkpoint improved over its chamber initialization on the
  randomized-fog test; median change was `+6.27 dB`.
- Every depth-dependent checkpoint improved on the depth-fog test; median
  change was `+5.13 dB`.
- On randomized fog, the depth-dependent checkpoints were lower than the
  randomized checkpoints for all 30 models; median difference was `-2.08 dB`.
- On depth fog, the depth-dependent checkpoints exceeded randomized
  checkpoints for all 30 models; median difference was `+3.41 dB`.
- Both synthetic branches lost chamber-domain performance. The median changes
  were `-5.24 dB` for randomized and `-5.36 dB` for depth-dependent training.

The complete 30-model tables are under `results/benchmark/`. NAFNet's exact
config and summary are under `results/nafnet/`.

## Important limitation

The depth-dependent NAFNet produced severe color clipping on five of six
audited raw aircraft frames. The synthetic metrics measure fit to their own
generators and are not real-fog benchmark results. For those reasons this
checkpoint is released as a research artifact, not as the default model.

## Directory layout

- `code/validated_benchmark/`: exact generator, ZoeDepth cache, and trainer used
  for the completed benchmark and released NAFNet.
- `code/synthetic_finetuning/`: the original GitHub prototype/GUI workflow
  preserved from the merged depth update.
- `results/`: compact NAFNet and 30-model evidence.
- `paper/`: corrected conservative depth-extension PDFs and buildable LaTeX.
- `weights/README.md`: matching Kaggle checkpoint and inference command.

Start with `code/validated_benchmark/README.md` to reproduce the released
checkpoint.
