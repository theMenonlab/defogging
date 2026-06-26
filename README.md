# Computational Defogging

This repository contains the code, small result tables, and reproduction notes for the computational defogging paper:

**From Fog Chamber to Aircraft Window: Pixel-Registered Imaging and Synthetic Fine-Tuning Enable Cross-Domain Defogging**

The project asks a practical question: can a model trained in a controlled fog chamber remove fog from images captured in different real-world settings? The released pipeline uses paired foggy/clear images from a display-based fog chamber, trains a NAFNet image-restoration model, then fine-tunes that model with randomized synthetic fog on clear outdoor images.

The large files are not stored in Git. Download model weights and datasets from Kaggle, then use this repository for the code and result tables.

## Download Links

| Asset | Link | Why it matters |
| --- | --- | --- |
| Model weights | https://www.kaggle.com/models/alingold/fog-removal | Released fog-chamber and synthetic fine-tuned NAFNet checkpoints |
| Fog-chamber dataset | https://www.kaggle.com/datasets/alingold/fog-chamber | Paired foggy/clear images used for the controlled restoration task |
| Synthetic fine-tuning source images | https://www.kaggle.com/datasets/kaggleprollc/mapillary-vistas-image-data-collection | Clear outdoor images used to synthesize randomized fog during fine-tuning |
| Source image archive for the chamber display | https://www.kaggle.com/datasets/rhtsingh/130k-images-512x512-universal-image-embeddings | Original 512 x 512 category images displayed in the fog chamber |

The model-weight Kaggle page should contain:

- `fog_chamber_nafnet_model_state_20260615.pth`
- `synthetic_finetuned_nafnet_model_state_20260615.pt`
- `run_config_fog_chamber_nafnet.json`
- `run_config_synthetic_finetuned_nafnet.json`
- `SHA256SUMS.txt`
- `checkpoints_manifest.csv`

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/theMenonlab/defogging.git
cd defogging
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If PyTorch installation fails, install the PyTorch build for your system from https://pytorch.org/get-started/locally/, then run `pip install -r requirements.txt` again.

### 2. Put the downloaded model files in one folder

Example local layout:

```text
release-assets/
  fog_chamber_nafnet_model_state_20260615.pth
  synthetic_finetuned_nafnet_model_state_20260615.pt
  run_config_fog_chamber_nafnet.json
  run_config_synthetic_finetuned_nafnet.json
  SHA256SUMS.txt
```

Check the downloaded weights if `SHA256SUMS.txt` is present:

```bash
cd release-assets
sha256sum -c SHA256SUMS.txt
cd ..
```

### 3. Run defogging on one image

Use the synthetic fine-tuned model for ordinary outdoor, aircraft-window, or real-world fog examples:

```bash
python code/nafnet_finetuning/run_defogging_inference.py \
  --checkpoint release-assets/synthetic_finetuned_nafnet_model_state_20260615.pt \
  --model-config release-assets/run_config_synthetic_finetuned_nafnet.json \
  --input path/to/foggy_image.jpg \
  --output-dir outputs/my_defogged_image \
  --save-comparison
```

Use the fog-chamber model when evaluating the controlled fog-chamber paired dataset:

```bash
python code/nafnet_finetuning/run_defogging_inference.py \
  --checkpoint release-assets/fog_chamber_nafnet_model_state_20260615.pth \
  --model-config release-assets/run_config_fog_chamber_nafnet.json \
  --input path/to/fog_chamber_images \
  --output-dir outputs/fog_chamber_predictions \
  --save-comparison
```

The `--input` path can be one image or a folder of images. Outputs are written as `*_defogged.png`; side-by-side previews are written as `*_comparison.jpg`.

## Expected Data Layout

For paired fog-chamber work, organize the downloaded dataset like this:

```text
data/
  fog_chamber/
    foggy/
      apparel/image0000.jpg
      cars/image0000.jpg
      ...
    ground_truth_matched/
      apparel/image0000.jpg
      cars/image0000.jpg
      ...
```

The important rule is that foggy and clear files must match by category and filename. For example:

```text
foggy/cars/image0390.jpg
ground_truth_matched/cars/image0390.jpg
```

The paper split uses every 10th image within each sorted category as the held-out test set. The matched fog-chamber set contains 5,495 paired images across six categories, with 552 held out for testing.

## Reproducing Paper Outputs

The small result tables are under `results/`. They are safe to keep in Git; the folder is about 5.8 MB even though it has more than 100 files.

Useful starting points:

- `results/latest_results_summary.json`: downstream NAFNet summary numbers.
- `results/fog_chamber_benchmark/benchmark_summary_table.csv`: 30-model benchmark summary.
- `results/supplement_tables/`: tables used in the supplement.
- `results/nafnet_runs/`: released NAFNet run summaries, configs, split manifests, and histories.

To commit `results/` through the command line instead of GitHub's web uploader:

```bash
git status --short
git add results
git commit -m "Add computational defogging result tables"
git push origin main
```

GitHub's browser uploader may reject folders with more than 100 files. The Git command-line path handles this normally.

## Training and Evaluation Workflows

### Fog-chamber NAFNet and inference

Core code lives in `code/nafnet_finetuning/`.

- `run_defogging_inference.py`: easiest public inference entrypoint.
- `infer_nafnet_fog.py`: older directory-oriented inference script kept for provenance.
- `train_nafnet_fog.py`: synthetic fog training utility.
- `train_real_haze_nafnet.py` and `train_ntire_supervised_nafnet.py`: task-specific paired-haze checks.

### Synthetic fine-tuning

Core code lives in `code/synthetic_finetuning/`.

The paper-current synthetic branch starts from the fog-chamber NAFNet checkpoint and fine-tunes on Mapillary Vistas clear images with spatial synthetic fog generated on the fly. This is a GPU workflow.

### Full 30-model benchmark

Core wrapper code lives in `code/fog_chamber_benchmark/`.

The 30-model benchmark requires the third-party model source trees used by the `run6` model roster. Those large third-party sources and checkpoints are not bundled here. The wrapper is included so the paired-data handling, training loop, metrics, and result format are transparent.

## Repository Layout

- `code/nafnet_finetuning/`: NAFNet training, inference, checkpoint, and evaluation utilities.
- `code/synthetic_finetuning/`: spatial synthetic-fog generation and synthetic fine-tuning workflow.
- `code/fog_chamber_benchmark/`: fog-chamber benchmark wrapper for paired fog/clear images.
- `code/classification_semantic_preservation/`: semantic-preservation classification checks.
- `code/dark_channel_prior/`: dark-channel-prior baseline.
- `code/fog_statistics/`: fog statistics and paired-image structure analyses.
- `results/`: small CSV/JSON result tables and split manifests.
- `paper/`: main/supplement PDFs and clean LaTeX source folders.
- `models/`: checkpoint manifests, checksums, and model config JSONs.
- `data/`: expected dataset layout and Kaggle notes.
- `docs/`: release upload and audit notes.

## Troubleshooting

- `ModuleNotFoundError: No module named 'torch'`: install PyTorch first, then reinstall `requirements.txt`.
- CUDA out of memory: use `--tile-size 256` or run with `--device cpu` for small tests.
- No images found: check that `--input` points to an image file or a folder containing image files.
- Wrong checkpoint/config pair: use `fog_chamber_nafnet_model_state_20260615.pth` with `run_config_fog_chamber_nafnet.json`, and use `synthetic_finetuned_nafnet_model_state_20260615.pt` with `run_config_synthetic_finetuned_nafnet.json`.
- Browser upload fails on GitHub: use `git add`, `git commit`, and `git push` from the command line.

## License

Code in this repository is released under the MIT License. Third-party datasets, third-party model implementations, and Kaggle-hosted assets have their own licenses and are not bundled here.
