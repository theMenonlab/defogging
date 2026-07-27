# Mixed chamber and synthetic training extension

Status: experimental post-submission extension. This method and checkpoint are
not part of the final PNAS submission.

This experiment trains one NAFNet from fresh initialization while
deterministically interleaving paired fog-chamber updates with randomized
synthetic Mapillary updates. It replaces the earlier sequential
chamber-then-synthetic schedule for this experimental branch only.

## Primary matched-budget model

- Chamber updates: `247,150` (`4,943` training pairs x `50` epochs)
- Randomized-synthetic updates: `2,600`
- Synthetic fraction of all updates: `1.041%`
- One chamber pair per chamber update
- Two synthetic pairs accumulated per synthetic update
- AdamW, learning rate `1e-4`, weight decay `1e-4`, seed 734
- Chamber loss: L1
- Synthetic loss: L1 + `0.02` mean-color + `0.008` residual-TV
- Output wrapper: `sigmoid_rgb`

The schedule uses integer error accumulation to spread exactly 2,600 synthetic
updates across the complete chamber-training sequence.

## Completed ratio sensitivity

| Schedule | Synthetic updates | Chamber PSNR / SSIM | Synthetic test PSNR / SSIM |
|---|---:|---:|---:|
| 0.2x | 520 | 24.237 / 0.7928 | 16.315 / 0.7255 |
| Matched budget | 2,600 | 24.408 / 0.7936 | 19.624 / 0.7909 |
| 5x | 13,000 | 24.440 / 0.7943 | 21.894 / 0.8407 |

The matched-budget model is the released checkpoint because it preserves the
synthetic exposure used by the sequential experiment. The 0.2x and 5x models
are sensitivity checks. Higher synthetic-test scores do not by themselves
establish better real-fog transfer.

## Directory layout

- `code/`: exact mixed trainer, randomized generator, minimal NAFNet benchmark
  source, and generic Slurm scripts.
- `results/`: sanitized run configs, summaries, and expanded comparison
  metrics for all three completed ratios.
- `paper/`: July 26 mixed-training paper variant and buildable LaTeX sources.
- `weights/README.md`: matching Kaggle checkpoint and inference command.

See `code/README.md` for smoke and full training commands.
