# Five-Class Semantic Preservation Check

Date: 2026-06-06

## Command

```bash
python run_five_class_semantic_preservation.py --overwrite-restored
```

## Design

- Classes retained: apparel, artwork, cars, dishes, furniture.
- Class removed: illustrations.
- Classifier: one ImageNet-initialized ResNet-50 trained only on clear target images.
- Test conditions for the same frozen classifier: clear targets, foggy inputs, NAFNet-restored foggy inputs.
- Split: same fog-chamber benchmark held-out IDs after excluding illustrations.
- Test set size: 460 images, 92 per class.

## Main Results

| Test input to frozen clear-trained classifier | Accuracy | Macro-F1 | Top-2 accuracy |
|---|---:|---:|---:|
| Clear targets | 98.91% | 98.91% | 100.00% |
| Foggy inputs | 82.83% | 83.46% | 95.65% |
| NAFNet-restored outputs | 94.13% | 94.16% | 97.83% |

NAFNet restoration improved recognition relative to foggy inputs for this clear-trained five-class classifier, but did not recover the clear-target ceiling.

Paired correctness tests:

| Comparison | Better for second condition | Worse for second condition | Exact McNemar p |
|---|---:|---:|---:|
| foggy inputs vs. NAFNet-restored outputs | 64 | 12 | 1.00e-09 |
| clear targets vs. NAFNet-restored outputs | 1 | 23 | 2.98e-06 |
| clear targets vs. foggy inputs | 3 | 77 | 1.41e-19 |

## Restoration Sanity Check

NAFNet-restored outputs against clear targets on the same five-class held-out split:

- PSNR: 24.50 dB
- SSIM: 0.7914
- MAE: 0.0493
- n: 460

## Output Files

Bulky per-image outputs are not staged in this GitHub package. The original run produced:

- `results/summary.json`
- `results/classification_summary_table.csv`
- `results/paired_correctness_tests.csv`
- `results/restored_vs_gt_summary.json`
- `results/frozen_clear_classifier_evaluations/*/classification_report.csv`
- `results/frozen_clear_classifier_evaluations/*/confusion_matrix.csv`
- `results/restored_images_png/`
