# Common evaluation report: `mixed_full_1024`

Completed units: 6 across 1 model(s).

## nafnet_fc

### nhaze

| Arm | n | Input PSNR | Output PSNR | SSIM | MAE | Delta PSNR |
|---|---:|---:|---:|---:|---:|---:|
| mixed_current | 55 | 11.403 | 12.903 | 0.4788 | 0.1831 | 1.500 |
| mixed_5x | 55 | 11.403 | 12.958 | 0.5071 | 0.1819 | 1.556 |

Paired PSNR differences (left minus right; 95% image-bootstrap CI):

- mixed_5x - mixed_current: 0.055 dB [-0.102, 0.208], n=55

### ntire_train

| Arm | n | Input PSNR | Output PSNR | SSIM | MAE | Delta PSNR |
|---|---:|---:|---:|---:|---:|---:|
| mixed_current | 25 | 14.848 | 18.369 | 0.7335 | 0.0966 | 3.521 |
| mixed_5x | 25 | 14.848 | 18.533 | 0.7516 | 0.0935 | 3.686 |

Paired PSNR differences (left minus right; 95% image-bootstrap CI):

- mixed_5x - mixed_current: 0.164 dB [-0.238, 0.572], n=25

### ohaze

| Arm | n | Input PSNR | Output PSNR | SSIM | MAE | Delta PSNR |
|---|---:|---:|---:|---:|---:|---:|
| mixed_current | 45 | 13.675 | 17.061 | 0.6675 | 0.1158 | 3.385 |
| mixed_5x | 45 | 13.675 | 15.517 | 0.6497 | 0.1394 | 1.841 |

Paired PSNR differences (left minus right; 95% image-bootstrap CI):

- mixed_5x - mixed_current: -1.544 dB [-1.730, -1.354], n=45

## Interpretation guardrails

- Cross-arm claims use paired differences on identical inputs.
- Public datasets are direct-transfer evaluations; none of these four checkpoints was trained on them.
- The sequential and mixed runs use the same randomized synthetic-fog generator.
- Mixed-current preserves the sequential run's 2,600 synthetic optimizer updates while interleaving them with 247,150 chamber updates.
- Mixed-5x keeps the chamber updates fixed and uses 13,000 synthetic updates.
- A confidence interval excluding zero is statistical evidence for this fixed image set, not proof of broad real-world superiority.
