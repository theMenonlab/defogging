#!/usr/bin/env python3
"""Evaluate the NTIRE input-only pseudo-finetuned model on no-reference proxies."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import benchmark_real_fog as bench

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="nafnet_ntire26_pseudo_mixedteacher_ft_20260601")
    parser.add_argument("--model-key", default="ntire26_pseudo_ft")
    parser.add_argument("--model-label", default="NTIRE pseudo-FT")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=int, default=96)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_pseudo_model(run_name: str) -> torch.nn.Module:
    spec = bench.ModelSpec(
        key="ntire26_pseudo_ft",
        label="Ours: NTIRE input-only pseudo-finetuned NAFNet",
        checkpoint=ROOT / "trained_models" / run_name / "checkpoints" / "best_model.pt",
        run_config=ROOT / "trained_models" / run_name / "run_config.json",
        note="Synthetic-fog checkpoint pseudo-finetuned on NTIRE inputs using mixed real-haze teacher outputs.",
    )
    return bench.load_model(spec)


def ntire_inputs() -> dict[str, list[Path]]:
    return {
        "NTIRE26-NH-train": sorted((DATA_DIR / "ntire26_train_inputs").glob("*.png")),
        "NTIRE26-NH-val": sorted((DATA_DIR / "ntire26_val_inputs").glob("*.png")),
        "NTIRE26-NH-test": sorted((DATA_DIR / "ntire26_test_inputs").glob("*.png")),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def panel(rgb: np.ndarray, title: str, subtitle: str, cell_w: int = 430, image_h: int = 260) -> Image.Image:
    image = Image.fromarray(np.clip(rgb * 255, 0, 255).astype(np.uint8))
    scale = min(cell_w / image.width, image_h / image.height)
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    out = Image.new("RGB", (cell_w, image_h + 74), "white")
    out.paste(resized, ((cell_w - resized.width) // 2, 74 + (image_h - resized.height) // 2))
    draw = ImageDraw.Draw(out)
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
        sub_font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        title_font = sub_font = ImageFont.load_default()
    draw.text((8, 7), title, fill=(0, 0, 0), font=title_font)
    draw.text((8, 40), subtitle, fill=(35, 35, 35), font=sub_font)
    return out


def write_grid(dataset: str, image_id: str, input_path: Path, out_path: Path, model_key: str) -> None:
    model_paths = [
        ("Hazy input", input_path),
        ("Synthetic FT", OUTPUT_DIR / "synthetic_ft" / dataset / f"{image_id}.png"),
        ("Variable synth", OUTPUT_DIR / "variable_synth" / dataset / f"{image_id}.png"),
        ("Mixed real-haze", OUTPUT_DIR / "real_mixed_ft" / dataset / f"{image_id}.png"),
        ("NTIRE pseudo-FT", OUTPUT_DIR / model_key / dataset / f"{image_id}.png"),
    ]
    panels = []
    for title, path in model_paths:
        rgb = bench.read_rgb(path)
        stats = bench.proxy_stats(rgb)
        subtitle = f"contrast {stats['rms_contrast']:.3f} / dark {stats['dark_channel']:.3f}"
        panels.append(panel(rgb, title, subtitle))
    pad = 12
    canvas = Image.new("RGB", (sum(p.width for p in panels) + pad * (len(panels) - 1), max(p.height for p in panels)), "white")
    x = 0
    for item in panels:
        canvas.paste(item, (x, 0))
        x += item.width + pad
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    model = load_pseudo_model(args.run_name)
    for dataset, paths in ntire_inputs().items():
        for path in tqdm(paths, desc=f"infer {dataset}"):
            out_path = OUTPUT_DIR / args.model_key / dataset / f"{path.stem}.png"
            if out_path.exists() and not args.overwrite:
                continue
            pred = bench.tiled_inference(model, bench.read_rgb(path), args.tile_size, args.tile_overlap)
            bench.save_rgb(out_path, pred)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    keys = [
        ("hazy_input", "Hazy input"),
        ("synthetic_ft", "Synthetic FT"),
        ("variable_synth", "Variable synth"),
        ("real_mixed_ft", "Mixed real-haze"),
        (args.model_key, args.model_label),
    ]
    rows: list[dict[str, object]] = []
    for dataset, paths in ntire_inputs().items():
        for path in paths:
            for key, label in keys:
                image_path = path if key == "hazy_input" else OUTPUT_DIR / key / dataset / f"{path.stem}.png"
                if not image_path.exists():
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "model_key": key,
                        "model_label": label,
                        "image_id": path.stem,
                        **bench.proxy_stats(bench.read_rgb(image_path)),
                    }
                )
    write_csv(
        RESULTS_DIR / "ntire26_pseudo_ft_proxy_metrics.csv",
        rows,
        ["dataset", "model_key", "model_label", "image_id", "dark_channel", "rms_contrast", "sharpness", "mean_luminance"],
    )

    summary_rows = []
    for dataset in sorted({r["dataset"] for r in rows}):
        for key, label in keys:
            subset = [r for r in rows if r["dataset"] == dataset and r["model_key"] == key]
            if not subset:
                continue
            summary_rows.append(
                {
                    "dataset": dataset,
                    "model_key": key,
                    "model_label": label,
                    "n": len(subset),
                    "mean_dark_channel": float(np.mean([r["dark_channel"] for r in subset])),
                    "mean_rms_contrast": float(np.mean([r["rms_contrast"] for r in subset])),
                    "mean_sharpness": float(np.mean([r["sharpness"] for r in subset])),
                    "mean_luminance": float(np.mean([r["mean_luminance"] for r in subset])),
                }
            )
    write_csv(
        RESULTS_DIR / "ntire26_pseudo_ft_proxy_summary.csv",
        summary_rows,
        [
            "dataset",
            "model_key",
            "model_label",
            "n",
            "mean_dark_channel",
            "mean_rms_contrast",
            "mean_sharpness",
            "mean_luminance",
        ],
    )

    examples = [
        ("NTIRE26-NH-train", "01_NTHazy"),
        ("NTIRE26-NH-val", "26_NTHazy"),
        ("NTIRE26-NH-test", "31_NTHazy"),
    ]
    for dataset, image_id in examples:
        split_name = dataset.lower()
        input_dir = {
            "NTIRE26-NH-train": DATA_DIR / "ntire26_train_inputs",
            "NTIRE26-NH-val": DATA_DIR / "ntire26_val_inputs",
            "NTIRE26-NH-test": DATA_DIR / "ntire26_test_inputs",
        }[dataset]
        write_grid(dataset, image_id, input_dir / f"{image_id}.png", FIGURES_DIR / f"{split_name}_pseudo_ft_grid.jpg", args.model_key)

    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_name": args.run_name,
        "model_key": args.model_key,
        "model_label": args.model_label,
        "caveat": "No NTIRE ground truth was available locally; metrics are no-reference proxies only.",
    }
    with (RESULTS_DIR / "ntire26_pseudo_ft_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    main()
