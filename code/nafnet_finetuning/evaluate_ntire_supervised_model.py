#!/usr/bin/env python3
"""Evaluate the supervised NTIRE 2026 fine-tuned model."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
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
    parser.add_argument("--run-name", default="nafnet_ntire26_supervised_ft_20260601")
    parser.add_argument("--model-key", default="ntire26_supervised_ft")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=int, default=96)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_model(run_name: str) -> torch.nn.Module:
    spec = bench.ModelSpec(
        key="ntire26_supervised_ft",
        label="Ours: NTIRE supervised NAFNet",
        checkpoint=ROOT / "trained_models" / run_name / "checkpoints" / "best_model.pt",
        run_config=ROOT / "trained_models" / run_name / "run_config.json",
        note="Synthetic-fog checkpoint supervised-finetuned on NTIRE 2026 train GT pairs.",
    )
    return bench.load_model(spec)


def input_splits() -> dict[str, list[Path]]:
    return {
        "NTIRE26-NH-train": sorted((DATA_DIR / "ntire26_train_inputs").glob("*.png")),
        "NTIRE26-NH-val": sorted((DATA_DIR / "ntire26_val_inputs").glob("*.png")),
        "NTIRE26-NH-test": sorted((DATA_DIR / "ntire26_test_inputs").glob("*.png")),
    }


def gt_for_train_input(path: Path) -> Path:
    image_id = path.stem.replace("_NTHazy", "")
    return DATA_DIR / "ntire26_train_gt" / f"{image_id}_GT.png"


def split_name(image_id: str) -> str:
    number = int(image_id.replace("_NTHazy", ""))
    return "train" if number <= 20 else "val" if number <= 23 else "test"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]], group_fields: list[str], metrics: list[str]) -> list[dict[str, object]]:
    out = []
    groups = sorted({tuple(r[f] for f in group_fields) for r in rows})
    for group in groups:
        subset = [r for r in rows if tuple(r[f] for f in group_fields) == group]
        row = {field: value for field, value in zip(group_fields, group)}
        row["n"] = len(subset)
        for metric in metrics:
            row[f"mean_{metric}"] = float(np.mean([r[metric] for r in subset]))
        out.append(row)
    return out


def panel(rgb: np.ndarray, title: str, subtitle: str, cell_w: int = 390, image_h: int = 260) -> Image.Image:
    image = Image.fromarray(np.clip(rgb * 255, 0, 255).astype(np.uint8))
    scale = min(cell_w / image.width, image_h / image.height)
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    out = Image.new("RGB", (cell_w, image_h + 74), "white")
    out.paste(resized, ((cell_w - resized.width) // 2, 74 + (image_h - resized.height) // 2))
    draw = ImageDraw.Draw(out)
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 23)
        sub_font = ImageFont.truetype("DejaVuSans.ttf", 17)
    except OSError:
        title_font = sub_font = ImageFont.load_default()
    draw.text((8, 7), title, fill=(0, 0, 0), font=title_font)
    draw.text((8, 40), subtitle, fill=(35, 35, 35), font=sub_font)
    return out


def save_grid(panels: list[Image.Image], out_path: Path) -> None:
    pad = 12
    canvas = Image.new("RGB", (sum(p.width for p in panels) + pad * (len(panels) - 1), max(p.height for p in panels)), "white")
    x = 0
    for item in panels:
        canvas.paste(item, (x, 0))
        x += item.width + pad
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)


def stack_images(rows: list[tuple[str, Path]], out_path: Path) -> None:
    images = [(label, Image.open(path).convert("RGB")) for label, path in rows]
    label_w = 210
    pad = 14
    width = max(image.width for _, image in images) + label_w
    height = sum(image.height for _, image in images) + pad * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        label_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 34)
    except OSError:
        label_font = ImageFont.load_default()
    y = 0
    for label, image in images:
        draw.text((12, y + 16), label, fill=(0, 0, 0), font=label_font)
        canvas.paste(image, (label_w, y))
        y += image.height + pad
    canvas.save(out_path, quality=95)


def save_labeled_rows(rows: list[tuple[str, list[Image.Image]]], out_path: Path) -> None:
    label_w = 150
    pad_x = 14
    pad_y = 16
    row_widths = [sum(panel.width for panel in panels) + pad_x * (len(panels) - 1) for _, panels in rows]
    row_heights = [max(panel.height for panel in panels) for _, panels in rows]
    canvas = Image.new(
        "RGB",
        (label_w + max(row_widths), sum(row_heights) + pad_y * (len(rows) - 1)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    try:
        label_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 36)
    except OSError:
        label_font = ImageFont.load_default()
    y = 0
    for (label, panels), row_h in zip(rows, row_heights):
        draw.text((8, y + 18), label, fill=(0, 0, 0), font=label_font)
        x = label_w
        for item in panels:
            canvas.paste(item, (x, y))
            x += item.width + pad_x
        y += row_h + pad_y
    canvas.save(out_path, quality=95)


def make_supervised_summary_figure(summary_rows: list[dict[str, object]]) -> None:
    order = [
        ("hazy_input", "Hazy\ninput"),
        ("synthetic_ft", "Synthetic\nFT"),
        ("variable_synth", "Variable\nsynth"),
        ("real_mixed_ft", "Mixed\nreal-haze"),
        ("ntire26_supervised_ft", "NTIRE\nsupervised"),
    ]
    split_rows = {r["model_key"]: r for r in summary_rows if r["split"] == "test"}
    labels = [label for key, label in order if key in split_rows]
    psnr = [float(split_rows[key]["mean_psnr"]) for key, _ in order if key in split_rows]
    ssim = [float(split_rows[key]["mean_ssim"]) for key, _ in order if key in split_rows]
    colors = ["#9aa1aa", "#6f8fb9", "#6aa78f", "#bd8a48", "#2f7d4f"][: len(labels)]

    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.2), constrained_layout=True)
    for ax, values, ylabel, ylim in [
        (axes[0], psnr, "PSNR (dB)", (0, max(psnr) + 3.0)),
        (axes[1], ssim, "SSIM", (0, 1.0)),
    ]:
        bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.45)
        bars[-1].set_linewidth(1.5)
        bars[-1].set_edgecolor("#0b3d20")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", color="#d7d7d7", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=8)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.35 if ylabel.startswith("PSNR") else 0.018),
                f"{value:.2f}" if ylabel.startswith("PSNR") else f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold" if bar is bars[-1] else "normal",
            )
    axes[0].set_title("Held-out NTIRE train-GT IDs 24--25")
    axes[1].set_title("Held-out NTIRE train-GT IDs 24--25")
    for suffix in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"ntire26_supervised_local_summary.{suffix}", dpi=220)
    plt.close(fig)


def metric_subtitle(pred: np.ndarray, gt: np.ndarray) -> str:
    vals = bench.metrics(pred, gt)
    return f"{vals['psnr']:.2f} dB / SSIM {vals['ssim']:.3f}"


def proxy_subtitle(rgb: np.ndarray) -> str:
    vals = bench.proxy_stats(rgb)
    return f"contrast {vals['rms_contrast']:.3f} / dark {vals['dark_channel']:.3f}"


def make_gt_grid(image_id: str, rows: list[dict[str, object]]) -> None:
    input_path = DATA_DIR / "ntire26_train_inputs" / f"{image_id}_NTHazy.png"
    gt_path = DATA_DIR / "ntire26_train_gt" / f"{image_id}_GT.png"
    gt = bench.read_rgb(gt_path)
    items = [
        ("Hazy input", input_path),
        ("Synthetic FT", OUTPUT_DIR / "synthetic_ft" / "NTIRE26-NH-train" / f"{image_id}_NTHazy.png"),
        ("Variable synth", OUTPUT_DIR / "variable_synth" / "NTIRE26-NH-train" / f"{image_id}_NTHazy.png"),
        ("Mixed real-haze", OUTPUT_DIR / "real_mixed_ft" / "NTIRE26-NH-train" / f"{image_id}_NTHazy.png"),
        ("NTIRE supervised", OUTPUT_DIR / "ntire26_supervised_ft" / "NTIRE26-NH-train" / f"{image_id}_NTHazy.png"),
    ]
    panels = []
    for title, path in items:
        rgb = bench.read_rgb(path)
        panels.append(panel(rgb, title, metric_subtitle(rgb, gt)))
    panels.append(panel(gt, "Ground truth", "reference"))
    save_grid(panels, FIGURES_DIR / f"ntire26-nh-localtest_{image_id}_supervised_grid.jpg")


def make_gt_key_pair() -> None:
    rows = []
    for image_id in ["24", "25"]:
        input_path = DATA_DIR / "ntire26_train_inputs" / f"{image_id}_NTHazy.png"
        gt_path = DATA_DIR / "ntire26_train_gt" / f"{image_id}_GT.png"
        gt = bench.read_rgb(gt_path)
        items = [
            ("Hazy input", input_path),
            ("Mixed real-haze", OUTPUT_DIR / "real_mixed_ft" / "NTIRE26-NH-train" / f"{image_id}_NTHazy.png"),
            ("NTIRE supervised", OUTPUT_DIR / "ntire26_supervised_ft" / "NTIRE26-NH-train" / f"{image_id}_NTHazy.png"),
            ("Ground truth", gt_path),
        ]
        panels = []
        for title, path in items:
            rgb = bench.read_rgb(path)
            subtitle = "reference" if title == "Ground truth" else metric_subtitle(rgb, gt)
            panels.append(panel(rgb, title, subtitle, cell_w=520, image_h=330))
        rows.append((f"ID {image_id}", panels))
    save_labeled_rows(rows, FIGURES_DIR / "ntire26-nh-localtest_supervised_key_pair.jpg")


def make_unpaired_grid(dataset: str, image_id: str) -> None:
    input_path = {
        "NTIRE26-NH-val": DATA_DIR / "ntire26_val_inputs",
        "NTIRE26-NH-test": DATA_DIR / "ntire26_test_inputs",
    }[dataset] / f"{image_id}.png"
    items = [
        ("Hazy input", input_path),
        ("Synthetic FT", OUTPUT_DIR / "synthetic_ft" / dataset / f"{image_id}.png"),
        ("Variable synth", OUTPUT_DIR / "variable_synth" / dataset / f"{image_id}.png"),
        ("Mixed real-haze", OUTPUT_DIR / "real_mixed_ft" / dataset / f"{image_id}.png"),
        ("NTIRE supervised", OUTPUT_DIR / "ntire26_supervised_ft" / dataset / f"{image_id}.png"),
    ]
    panels = []
    for title, path in items:
        rgb = bench.read_rgb(path)
        panels.append(panel(rgb, title, proxy_subtitle(rgb)))
    save_grid(panels, FIGURES_DIR / f"{dataset.lower()}_supervised_ft_grid.jpg")


def make_unpaired_key_pair() -> None:
    rows = []
    specs = [
        ("Val 26", "NTIRE26-NH-val", DATA_DIR / "ntire26_val_inputs", "26_NTHazy"),
        ("Test 31", "NTIRE26-NH-test", DATA_DIR / "ntire26_test_inputs", "31_NTHazy"),
    ]
    for label, dataset, input_dir, image_id in specs:
        items = [
            ("Hazy input", input_dir / f"{image_id}.png"),
            ("Synthetic FT", OUTPUT_DIR / "synthetic_ft" / dataset / f"{image_id}.png"),
            ("Mixed real-haze", OUTPUT_DIR / "real_mixed_ft" / dataset / f"{image_id}.png"),
            ("NTIRE supervised", OUTPUT_DIR / "ntire26_supervised_ft" / dataset / f"{image_id}.png"),
        ]
        panels = []
        for title, path in items:
            rgb = bench.read_rgb(path)
            panels.append(panel(rgb, title, proxy_subtitle(rgb), cell_w=520, image_h=330))
        rows.append((label, panels))
    save_labeled_rows(rows, FIGURES_DIR / "ntire26-nh-hidden_supervised_key_pair.jpg")


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    model = load_model(args.run_name)
    for dataset, paths in input_splits().items():
        for path in tqdm(paths, desc=f"infer {dataset}"):
            out_path = OUTPUT_DIR / args.model_key / dataset / f"{path.stem}.png"
            if out_path.exists() and not args.overwrite:
                continue
            pred = bench.tiled_inference(model, bench.read_rgb(path), args.tile_size, args.tile_overlap)
            bench.save_rgb(out_path, pred)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    paired_model_paths = {
        "hazy_input": ("Hazy input", lambda dataset, image_id: DATA_DIR / "ntire26_train_inputs" / f"{image_id}.png"),
        "synthetic_ft": ("Synthetic FT", lambda dataset, image_id: OUTPUT_DIR / "synthetic_ft" / dataset / f"{image_id}.png"),
        "variable_synth": ("Variable synth", lambda dataset, image_id: OUTPUT_DIR / "variable_synth" / dataset / f"{image_id}.png"),
        "real_mixed_ft": ("Mixed real-haze", lambda dataset, image_id: OUTPUT_DIR / "real_mixed_ft" / dataset / f"{image_id}.png"),
        "ntire26_supervised_ft": ("NTIRE supervised FT", lambda dataset, image_id: OUTPUT_DIR / args.model_key / dataset / f"{image_id}.png"),
    }
    paired_rows: list[dict[str, object]] = []
    dataset = "NTIRE26-NH-train"
    for path in input_splits()[dataset]:
        image_id = path.stem
        gt = bench.read_rgb(gt_for_train_input(path))
        for key, (label, make_path) in paired_model_paths.items():
            image_path = make_path(dataset, image_id)
            if not image_path.exists():
                continue
            pred = bench.read_rgb(image_path)
            paired_rows.append(
                {
                    "split": split_name(image_id),
                    "model_key": key,
                    "model_label": label,
                    "image_id": image_id,
                    **bench.metrics(pred, gt),
                }
            )
    write_csv(
        RESULTS_DIR / "ntire26_supervised_paired_metrics.csv",
        paired_rows,
        ["split", "model_key", "model_label", "image_id", "mae", "mse", "psnr", "ssim"],
    )
    summary_rows = summarize(paired_rows, ["split", "model_key", "model_label"], ["mae", "mse", "psnr", "ssim"])
    write_csv(
        RESULTS_DIR / "ntire26_supervised_paired_summary.csv",
        summary_rows,
        ["split", "model_key", "model_label", "n", "mean_mae", "mean_mse", "mean_psnr", "mean_ssim"],
    )
    make_supervised_summary_figure(summary_rows)

    proxy_rows: list[dict[str, object]] = []
    for dataset in ["NTIRE26-NH-val", "NTIRE26-NH-test"]:
        for path in input_splits()[dataset]:
            for key, label in [
                ("hazy_input", "Hazy input"),
                ("synthetic_ft", "Synthetic FT"),
                ("variable_synth", "Variable synth"),
                ("real_mixed_ft", "Mixed real-haze"),
                (args.model_key, "NTIRE supervised FT"),
            ]:
                image_path = path if key == "hazy_input" else OUTPUT_DIR / key / dataset / f"{path.stem}.png"
                if not image_path.exists():
                    continue
                proxy_rows.append(
                    {
                        "dataset": dataset,
                        "model_key": key,
                        "model_label": label,
                        "image_id": path.stem,
                        **bench.proxy_stats(bench.read_rgb(image_path)),
                    }
                )
    write_csv(
        RESULTS_DIR / "ntire26_supervised_unpaired_proxy_metrics.csv",
        proxy_rows,
        ["dataset", "model_key", "model_label", "image_id", "dark_channel", "rms_contrast", "sharpness", "mean_luminance"],
    )
    proxy_summary = summarize(proxy_rows, ["dataset", "model_key", "model_label"], ["dark_channel", "rms_contrast", "sharpness", "mean_luminance"])
    write_csv(
        RESULTS_DIR / "ntire26_supervised_unpaired_proxy_summary.csv",
        proxy_summary,
        ["dataset", "model_key", "model_label", "n", "mean_dark_channel", "mean_rms_contrast", "mean_sharpness", "mean_mean_luminance"],
    )

    make_gt_grid("24", paired_rows)
    make_gt_grid("25", paired_rows)
    make_unpaired_grid("NTIRE26-NH-val", "26_NTHazy")
    make_unpaired_grid("NTIRE26-NH-test", "31_NTHazy")
    make_gt_key_pair()
    make_unpaired_key_pair()
    stack_images(
        [
            ("ID 24", FIGURES_DIR / "ntire26-nh-localtest_24_supervised_grid.jpg"),
            ("ID 25", FIGURES_DIR / "ntire26-nh-localtest_25_supervised_grid.jpg"),
        ],
        FIGURES_DIR / "ntire26-nh-localtest_supervised_pair.jpg",
    )
    stack_images(
        [
            ("Val 26", FIGURES_DIR / "ntire26-nh-val_supervised_ft_grid.jpg"),
            ("Test 31", FIGURES_DIR / "ntire26-nh-test_supervised_ft_grid.jpg"),
        ],
        FIGURES_DIR / "ntire26-nh-hidden_supervised_pair.jpg",
    )

    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_name": args.run_name,
        "model_key": args.model_key,
        "caveat": "Only NTIRE train GT is available locally; validation/test GT remains hidden.",
    }
    with (RESULTS_DIR / "ntire26_supervised_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    main()
