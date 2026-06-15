#!/usr/bin/env python3
"""Evaluate a trained RGB NAFNet checkpoint on paired O-HAZE, NH-HAZE, and NTIRE images."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "general_code"
sys.path.insert(0, str(CODE_DIR))

from infer_nafnet_fog import build_model, tiled_inference  # noqa: E402
from train_real_haze_nafnet import collect_nhhaze, collect_ohaze  # noqa: E402
from train_ntire_supervised_nafnet import collect_ntire_pairs  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=int, default=96)
    parser.add_argument("--max-records-per-dataset", type=int, default=None)
    return parser.parse_args()


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def save_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype(np.uint8)).save(path)


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


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.clip(pred, 0, 1)
    target = np.clip(target, 0, 1)
    pred_color = downsample_for_color(pred)
    target_color = downsample_for_color(target)
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
        "mse": float(np.mean((pred - target) ** 2)),
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


def resize_like(arr: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    if arr.shape == shape:
        return arr
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype(np.uint8))
    im = im.resize(shape[1::-1], Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


def evenly_sample_records(records: list[object], count: int) -> list[object]:
    if len(records) <= count:
        return records
    return [records[round(i * (len(records) - 1) / (count - 1))] for i in range(count)]


def limit_records(records: list[object], max_records_per_dataset: int | None) -> list[object]:
    if max_records_per_dataset is None:
        return records
    limited: list[object] = []
    datasets = sorted({getattr(record, "dataset") for record in records})
    for dataset in datasets:
        subset = [record for record in records if getattr(record, "dataset") == dataset]
        limited.extend(evenly_sample_records(subset, max_records_per_dataset))
    return limited


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.run_config).open("r", encoding="utf-8") as handle:
        run_config = json.load(handle)
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model = build_model(run_config).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    records = limit_records(collect_ohaze() + collect_nhhaze() + collect_ntire_pairs(), args.max_records_per_dataset)
    rows: list[dict[str, object]] = []
    start = time.time()
    for record in tqdm(records, desc="paired public eval"):
        hazy = read_rgb(record.hazy_path)
        gt = resize_like(read_rgb(record.gt_path), hazy.shape)
        pred = tiled_inference(model, hazy, args.tile_size, args.tile_overlap)
        pred_name = f"{record.dataset}_{record.image_id}".replace("/", "_")
        pred_path = out_dir / "predictions" / f"{pred_name}.png"
        compare_path = out_dir / "comparisons" / f"{pred_name}_compare.jpg"
        save_rgb(pred_path, pred)
        save_rgb(compare_path, np.concatenate([hazy, pred, gt], axis=1))
        input_metrics = metrics(hazy, gt)
        pred_metrics = metrics(pred, gt)
        rows.append(
            {
                "dataset": record.dataset,
                "image_id": record.image_id,
                "split": record.split,
                "hazy_path": str(record.hazy_path),
                "gt_path": str(record.gt_path),
                "prediction_path": str(pred_path),
                "comparison_path": str(compare_path),
                "input_psnr": input_metrics["psnr"],
                "input_ssim": input_metrics["ssim"],
                "input_mae": input_metrics["mae"],
                "pred_psnr": pred_metrics["psnr"],
                "pred_ssim": pred_metrics["ssim"],
                "pred_mae": pred_metrics["mae"],
                "input_delta_e00": input_metrics["delta_e00"],
                "pred_delta_e00": pred_metrics["delta_e00"],
                "input_mean_r_bias": input_metrics["mean_r_bias"],
                "input_mean_g_bias": input_metrics["mean_g_bias"],
                "input_mean_b_bias": input_metrics["mean_b_bias"],
                "pred_mean_r_bias": pred_metrics["mean_r_bias"],
                "pred_mean_g_bias": pred_metrics["mean_g_bias"],
                "pred_mean_b_bias": pred_metrics["mean_b_bias"],
                "input_mean_sat_bias": input_metrics["mean_sat_bias"],
                "pred_mean_sat_bias": pred_metrics["mean_sat_bias"],
                "input_bright_mae": input_metrics["bright_mae"],
                "pred_bright_mae": pred_metrics["bright_mae"],
                "input_bright_delta_e00": input_metrics["bright_delta_e00"],
                "pred_bright_delta_e00": pred_metrics["bright_delta_e00"],
            }
        )

    csv_path = out_dir / "paired_public_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        subset = [row for row in rows if row["dataset"] == dataset]
        for split in ["all"] + sorted({str(row["split"]) for row in subset}):
            split_subset = subset if split == "all" else [row for row in subset if row["split"] == split]
            if not split_subset:
                continue
            summary_rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "n": len(split_subset),
                    "input_psnr": float(np.mean([float(row["input_psnr"]) for row in split_subset])),
                    "input_ssim": float(np.mean([float(row["input_ssim"]) for row in split_subset])),
                    "pred_psnr": float(np.mean([float(row["pred_psnr"]) for row in split_subset])),
                    "pred_ssim": float(np.mean([float(row["pred_ssim"]) for row in split_subset])),
                    "delta_psnr": float(np.mean([float(row["pred_psnr"]) - float(row["input_psnr"]) for row in split_subset])),
                    "delta_ssim": float(np.mean([float(row["pred_ssim"]) - float(row["input_ssim"]) for row in split_subset])),
                    "input_delta_e00": float(np.mean([float(row["input_delta_e00"]) for row in split_subset])),
                    "pred_delta_e00": float(np.mean([float(row["pred_delta_e00"]) for row in split_subset])),
                    "delta_delta_e00": float(np.mean([float(row["pred_delta_e00"]) - float(row["input_delta_e00"]) for row in split_subset])),
                    "pred_mean_r_bias": float(np.mean([float(row["pred_mean_r_bias"]) for row in split_subset])),
                    "pred_mean_g_bias": float(np.mean([float(row["pred_mean_g_bias"]) for row in split_subset])),
                    "pred_mean_b_bias": float(np.mean([float(row["pred_mean_b_bias"]) for row in split_subset])),
                    "pred_mean_sat_bias": float(np.mean([float(row["pred_mean_sat_bias"]) for row in split_subset])),
                    "input_bright_mae": float(np.nanmean([float(row["input_bright_mae"]) for row in split_subset])),
                    "pred_bright_mae": float(np.nanmean([float(row["pred_bright_mae"]) for row in split_subset])),
                    "input_bright_delta_e00": float(np.nanmean([float(row["input_bright_delta_e00"]) for row in split_subset])),
                    "pred_bright_delta_e00": float(np.nanmean([float(row["pred_bright_delta_e00"]) for row in split_subset])),
                }
            )

    with (out_dir / "paired_public_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    payload = {
        "checkpoint": args.checkpoint,
        "run_config": args.run_config,
        "output_dir": str(out_dir),
        "elapsed_seconds": time.time() - start,
        "summary": summary_rows,
    }
    with (out_dir / "paired_public_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
