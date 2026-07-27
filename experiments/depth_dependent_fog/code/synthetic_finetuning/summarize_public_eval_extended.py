#!/usr/bin/env python3
"""Add color and bright-region metrics to an existing paired public eval folder."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def resize_like(arr: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    if arr.shape == shape:
        return arr
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype(np.uint8))
    im = im.resize(shape[1::-1], Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


def downsample_for_color(arr: np.ndarray, max_side: int = 1024) -> np.ndarray:
    height, width = arr.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return arr
    scale = max_side / float(longest)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype(np.uint8))
    im = im.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


def resolve_prediction_path(path_text: str, eval_dir: Path) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    remapped = eval_dir / "predictions" / path.name
    if remapped.exists():
        return remapped
    raise FileNotFoundError(path_text)


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.clip(pred, 0, 1)
    target = np.clip(target, 0, 1)
    pred_color = downsample_for_color(pred)
    target_color = downsample_for_color(target)
    if pred_color.shape != target_color.shape:
        target_color = resize_like(target_color, pred_color.shape)
    mean_bias = pred_color.mean(axis=(0, 1)) - target_color.mean(axis=(0, 1))
    pred_lab = rgb2lab(pred_color)
    target_lab = rgb2lab(target_color)
    luminance = target_color.mean(axis=2)
    bright_mask = luminance > 0.72
    if np.any(bright_mask):
        bright_mae = float(np.mean(np.abs(pred_color[bright_mask] - target_color[bright_mask])))
        bright_delta_e = float(np.mean(deltaE_ciede2000(pred_lab[bright_mask], target_lab[bright_mask])))
    else:
        bright_mae = float("nan")
        bright_delta_e = float("nan")
    pred_sat = pred_color.max(axis=2) - pred_color.min(axis=2)
    target_sat = target_color.max(axis=2) - target_color.min(axis=2)
    return {
        "mae": float(np.mean(np.abs(pred - target))),
        "psnr": float(peak_signal_noise_ratio(target, pred, data_range=1.0)),
        "ssim": float(structural_similarity(target, pred, channel_axis=2, data_range=1.0)),
        "delta_e00": float(np.mean(deltaE_ciede2000(pred_lab, target_lab))),
        "mean_r_bias": float(mean_bias[0]),
        "mean_g_bias": float(mean_bias[1]),
        "mean_b_bias": float(mean_bias[2]),
        "mean_sat_bias": float(np.mean(pred_sat - target_sat)),
        "bright_mae": bright_mae,
        "bright_delta_e00": bright_delta_e,
    }


def mean_float(rows: list[dict[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return float(np.nanmean(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", required=True, type=Path)
    args = parser.parse_args()

    metrics_csv = args.eval_dir / "paired_public_metrics.csv"
    with metrics_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {metrics_csv}")

    out_rows: list[dict[str, object]] = []
    for row in rows:
        hazy = read_rgb(Path(row["hazy_path"]))
        gt = resize_like(read_rgb(Path(row["gt_path"])), hazy.shape)
        pred = resize_like(read_rgb(resolve_prediction_path(row["prediction_path"], args.eval_dir)), hazy.shape)
        input_metrics = metrics(hazy, gt)
        pred_metrics = metrics(pred, gt)
        out = dict(row)
        for prefix, payload in [("input", input_metrics), ("pred", pred_metrics)]:
            for key, value in payload.items():
                out[f"{prefix}_{key}"] = value
        out_rows.append(out)

    extended_csv = args.eval_dir / "paired_public_metrics_extended.csv"
    with extended_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    summary_rows: list[dict[str, object]] = []
    for dataset in sorted({str(row["dataset"]) for row in out_rows}):
        dataset_rows = [row for row in out_rows if row["dataset"] == dataset]
        for split in ["all"] + sorted({str(row["split"]) for row in dataset_rows}):
            subset = dataset_rows if split == "all" else [row for row in dataset_rows if row["split"] == split]
            if not subset:
                continue
            summary_rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "n": len(subset),
                    "input_psnr": mean_float(subset, "input_psnr"),
                    "input_ssim": mean_float(subset, "input_ssim"),
                    "pred_psnr": mean_float(subset, "pred_psnr"),
                    "pred_ssim": mean_float(subset, "pred_ssim"),
                    "delta_psnr": float(np.mean([float(row["pred_psnr"]) - float(row["input_psnr"]) for row in subset])),
                    "delta_ssim": float(np.mean([float(row["pred_ssim"]) - float(row["input_ssim"]) for row in subset])),
                    "input_delta_e00": mean_float(subset, "input_delta_e00"),
                    "pred_delta_e00": mean_float(subset, "pred_delta_e00"),
                    "delta_delta_e00": float(np.mean([float(row["pred_delta_e00"]) - float(row["input_delta_e00"]) for row in subset])),
                    "pred_mean_r_bias": mean_float(subset, "pred_mean_r_bias"),
                    "pred_mean_g_bias": mean_float(subset, "pred_mean_g_bias"),
                    "pred_mean_b_bias": mean_float(subset, "pred_mean_b_bias"),
                    "pred_mean_sat_bias": mean_float(subset, "pred_mean_sat_bias"),
                    "input_bright_mae": mean_float(subset, "input_bright_mae"),
                    "pred_bright_mae": mean_float(subset, "pred_bright_mae"),
                    "input_bright_delta_e00": mean_float(subset, "input_bright_delta_e00"),
                    "pred_bright_delta_e00": mean_float(subset, "pred_bright_delta_e00"),
                }
            )

    summary_csv = args.eval_dir / "paired_public_summary_extended.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(summary_csv)


if __name__ == "__main__":
    main()
