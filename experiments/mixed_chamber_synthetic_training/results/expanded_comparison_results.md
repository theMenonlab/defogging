# Expanded mixed-training comparison

This standalone diagnostic adds the exact Fig. 3 top-left aircraft input, a representative unpaired image from `Real_Fog_Test_20260523`, and the Fig. 5 `dishes_image0520` fog-chamber pair to the previous four O-HAZE/NH-HAZE rows. It has not been integrated into the paper.

The aircraft example has no aligned clear reference, so it is shown qualitatively without PSNR/SSIM. To keep inference practical while retaining more than the displayed figure resolution, the 3024x4032 source was resized to 1008x1344 before identical tiled inference with every checkpoint.

The selected real-fog frame (`outdoor_20260522_184817_00005.jpg`) is also unpaired and is therefore shown qualitatively. It was selected because the scene contains strong spatially varying fog together with buildings, trees, pavement, and foreground foliage. The 1920x1080 source was resized to 1280x720 before identical tiled inference with every checkpoint.

## Selected Fig. 5 fog-chamber pair

| Setting | PSNR | SSIM |
|---|---:|---:|
| Fog-chamber only | 22.23 dB | 0.678 |
| Prior sequential, 2,600 synthetic updates | 13.75 dB | 0.340 |
| Mixed 0.2x synthetic | 22.13 dB | 0.673 |
| Mixed current ratio | 21.80 dB | 0.668 |
| Mixed 5x synthetic | 22.05 dB | 0.672 |

The fog-chamber row is one selected paired example, not the 552-image aggregate.
