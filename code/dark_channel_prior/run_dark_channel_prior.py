#!/usr/bin/env python3
"""Evaluate a classical dark-channel-prior dehazing baseline on paired tests."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm


ROOT = Path(".")
PAPER_DIR = ROOT / "fog_imager_3/paper_figures"
DEFAULT_OUTPUT = ROOT / "classical_defogging/dcp_results"
TEST_SETS = {
    "No polarizer": ROOT
    / "phamscope_nafnet/experiments_fog_rgb_no_filt_medium_20260321/test_results/metrics.csv",
    "Vertical polarizer": ROOT
    / "phamscope_nafnet/experiments_fog_rgb_vert_filt_665_20260325/test_results/metrics.csv",
}
NAFNET_RESULTS = {
    label: csv_path.parent / "images" for label, csv_path in TEST_SETS.items()
}


def load_rgb(path: Path, size: tuple[int, int] = (512, 512)) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if image.size != size:
        image = image.resize(size, Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def min_filter(image: np.ndarray, patch_size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    return cv2.erode(image.astype(np.float32), kernel)


def estimate_airlight(image: np.ndarray, dark: np.ndarray, top_fraction: float) -> np.ndarray:
    flat_dark = dark.reshape(-1)
    n_pixels = flat_dark.size
    n_top = max(1, int(round(n_pixels * top_fraction)))
    candidate_idx = np.argpartition(flat_dark, -n_top)[-n_top:]
    flat_image = image.reshape(-1, 3)
    brightest = candidate_idx[np.argmax(flat_image[candidate_idx].sum(axis=1))]
    return np.maximum(flat_image[brightest], 1e-6)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Fast guided filter for single-channel guide/source arrays."""
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)
    win = (2 * radius + 1, 2 * radius + 1)
    mean_i = cv2.boxFilter(guide, -1, win, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_p = cv2.boxFilter(src, -1, win, normalize=True, borderType=cv2.BORDER_REFLECT)
    corr_i = cv2.boxFilter(guide * guide, -1, win, normalize=True, borderType=cv2.BORDER_REFLECT)
    corr_ip = cv2.boxFilter(guide * src, -1, win, normalize=True, borderType=cv2.BORDER_REFLECT)
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    mean_a = cv2.boxFilter(a, -1, win, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_b = cv2.boxFilter(b, -1, win, normalize=True, borderType=cv2.BORDER_REFLECT)
    return mean_a * guide + mean_b


def dark_channel_prior(
    image: np.ndarray,
    patch_size: int = 15,
    omega: float = 0.95,
    top_fraction: float = 0.001,
    guided_radius: int = 40,
    guided_eps: float = 1e-3,
    t0: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dark = min_filter(np.min(image, axis=2), patch_size)
    airlight = estimate_airlight(image, dark, top_fraction)
    normalized = image / airlight.reshape(1, 1, 3)
    raw_t = 1.0 - omega * min_filter(np.min(normalized, axis=2), patch_size)
    guide = cv2.cvtColor((image * 255).round().clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    guide = guide.astype(np.float32) / 255.0
    refined_t = guided_filter(guide, raw_t, guided_radius, guided_eps)
    refined_t = np.clip(refined_t, 0.0, 1.0)
    recovered = (image - airlight.reshape(1, 1, 3)) / np.maximum(refined_t[..., None], t0) + airlight.reshape(1, 1, 3)
    return np.clip(recovered, 0.0, 1.0), refined_t, airlight


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.clip(pred, 0.0, 1.0)
    target = np.clip(target, 0.0, 1.0)
    return {
        "mae": float(np.mean(np.abs(pred - target))),
        "mse": float(np.mean((pred - target) ** 2)),
        "psnr": float(peak_signal_noise_ratio(target, pred, data_range=1.0)),
        "ssim": float(structural_similarity(target, pred, channel_axis=2, data_range=1.0)),
    }


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def nafnet_prediction_path(row: object) -> Path:
    base = f"{row.category}_{Path(row.image_name).stem}"
    image_dir = NAFNET_RESULTS[row.condition]
    for suffix in ("_pred.png", "_pred.tif"):
        candidate = image_dir / f"{base}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing NAFNet prediction for {row.condition}: {base}")


def evaluate_set(label: str, csv_path: Path, output_root: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    rows: list[dict[str, object]] = []
    set_slug = label.lower().replace(" ", "_")
    image_dir = output_root / set_slug / "images"
    for row in tqdm(df.itertuples(index=False), total=len(df), desc=label):
        input_path = Path(row.aligned_path)
        target_path = Path(row.gt_path)
        fog = load_rgb(input_path)
        target = load_rgb(target_path)
        start = time.time()
        pred, transmission, airlight = dark_channel_prior(
            fog,
            patch_size=args.patch_size,
            omega=args.omega,
            top_fraction=args.top_fraction,
            guided_radius=args.guided_radius,
            guided_eps=args.guided_eps,
            t0=args.t0,
        )
        elapsed = time.time() - start
        dcp_metrics = compute_metrics(pred, target)
        input_metrics = compute_metrics(fog, target)
        out_name = f"{row.category}_{Path(row.image_name).stem}"
        if args.save_outputs:
            save_rgb(image_dir / f"{out_name}_dcp.png", pred)
        rows.append(
            {
                "condition": label,
                "category": row.category,
                "image_name": row.image_name,
                "input_path": str(input_path),
                "gt_path": str(target_path),
                "dcp_output_path": str(image_dir / f"{out_name}_dcp.png") if args.save_outputs else "",
                "airlight_r": float(airlight[0]),
                "airlight_g": float(airlight[1]),
                "airlight_b": float(airlight[2]),
                "transmission_mean": float(np.mean(transmission)),
                "transmission_min": float(np.min(transmission)),
                "inference_seconds": elapsed,
                "input_mae": input_metrics["mae"],
                "input_mse": input_metrics["mse"],
                "input_psnr": input_metrics["psnr"],
                "input_ssim": input_metrics["ssim"],
                "dcp_mae": dcp_metrics["mae"],
                "dcp_mse": dcp_metrics["mse"],
                "dcp_psnr": dcp_metrics["psnr"],
                "dcp_ssim": dcp_metrics["ssim"],
            }
        )
    metrics_df = pd.DataFrame(rows)
    metrics_path = output_root / set_slug / "metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)

    overall = (
        metrics_df[
            [
                "input_mae",
                "input_mse",
                "input_psnr",
                "input_ssim",
                "dcp_mae",
                "dcp_mse",
                "dcp_psnr",
                "dcp_ssim",
                "inference_seconds",
                "transmission_mean",
            ]
        ]
        .mean()
        .to_dict()
    )
    overall["condition"] = label
    overall["samples"] = int(len(metrics_df))
    category = (
        metrics_df.groupby(["condition", "category"])[
            ["input_mae", "input_mse", "input_psnr", "input_ssim", "dcp_mae", "dcp_mse", "dcp_psnr", "dcp_ssim"]
        ]
        .mean()
        .reset_index()
    )
    return pd.DataFrame([overall]), category


def build_example_figure(output_root: Path, metrics_df: pd.DataFrame, figure_path: Path) -> None:
    ranked = metrics_df.copy()
    ranked["abs_rank"] = ranked.groupby("condition")["dcp_psnr"].transform(lambda s: (s - s.median()).abs())
    examples = ranked.sort_values(["condition", "abs_rank"]).groupby("condition", sort=False).head(1)
    fig, axes = plt.subplots(len(examples), 4, figsize=(9.4, 2.15 * len(examples)))
    if len(examples) == 1:
        axes = axes[None, :]
    for ax_row, row in zip(axes, examples.itertuples(index=False), strict=True):
        fog = load_rgb(Path(row.input_path))
        target = load_rgb(Path(row.gt_path))
        pred = load_rgb(Path(row.dcp_output_path)) if row.dcp_output_path else fog
        nafnet = load_rgb(nafnet_prediction_path(row))
        images = [fog, pred, nafnet, target]
        titles = ["Foggy input", "DCP output", "Paired NAFNet", "Clear target"]
        for ax, image, title in zip(ax_row, images, titles, strict=True):
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=9)
        ax_row[0].set_ylabel(f"{row.condition}\n{row.category} {Path(row.image_name).stem}", fontsize=8)
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--patch-size", type=int, default=15)
    parser.add_argument("--omega", type=float, default=0.95)
    parser.add_argument("--top-fraction", type=float, default=0.001)
    parser.add_argument("--guided-radius", type=int, default=40)
    parser.add_argument("--guided-eps", type=float, default=1e-3)
    parser.add_argument("--t0", type=float, default=0.10)
    parser.add_argument("--save-outputs", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    all_overall: list[pd.DataFrame] = []
    all_category: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []
    for label, csv_path in TEST_SETS.items():
        overall, category = evaluate_set(label, csv_path, args.output_root, args)
        all_overall.append(overall)
        all_category.append(category)
        all_metrics.append(pd.read_csv(args.output_root / label.lower().replace(" ", "_") / "metrics.csv"))

    overall_df = pd.concat(all_overall, ignore_index=True)[
        [
            "condition",
            "samples",
            "input_mae",
            "input_mse",
            "input_psnr",
            "input_ssim",
            "dcp_mae",
            "dcp_mse",
            "dcp_psnr",
            "dcp_ssim",
            "inference_seconds",
            "transmission_mean",
        ]
    ]
    category_df = pd.concat(all_category, ignore_index=True)
    metrics_df = pd.concat(all_metrics, ignore_index=True)
    overall_df.to_csv(args.output_root / "dark_channel_prior_overall.csv", index=False)
    category_df.to_csv(args.output_root / "dark_channel_prior_by_category.csv", index=False)
    metrics_df.to_csv(args.output_root / "dark_channel_prior_all_metrics.csv", index=False)
    build_example_figure(args.output_root, metrics_df, args.output_root / "dark_channel_prior_examples.pdf")

    manifest = {
        "method": "Dark channel prior dehazing",
        "reference": {
            "citation_key": "he2010single",
            "doi": "10.1109/TPAMI.2010.168",
            "bibtex": "@article{he2010single, author={He, Kaiming and Sun, Jian and Tang, Xiaoou}, title={Single Image Haze Removal Using Dark Channel Prior}, journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, volume={33}, number={12}, pages={2341--2353}, year={2011}, doi={10.1109/TPAMI.2010.168}}",
        },
        "parameters": {
            "patch_size": args.patch_size,
            "omega": args.omega,
            "top_fraction": args.top_fraction,
            "guided_radius": args.guided_radius,
            "guided_eps": args.guided_eps,
            "t0": args.t0,
            "save_outputs": args.save_outputs,
        },
        "test_csvs": {label: str(path) for label, path in TEST_SETS.items()},
        "outputs": {
            "overall": str(args.output_root / "dark_channel_prior_overall.csv"),
            "by_category": str(args.output_root / "dark_channel_prior_by_category.csv"),
            "all_metrics": str(args.output_root / "dark_channel_prior_all_metrics.csv"),
            "examples": str(args.output_root / "dark_channel_prior_examples.pdf"),
        },
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(overall_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
