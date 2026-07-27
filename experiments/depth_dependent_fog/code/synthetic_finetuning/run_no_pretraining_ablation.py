#!/usr/bin/env python3
"""Run the synthetic fine-tuning ablation without fog-chamber pretraining."""

from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from run_overnight_synthetic_fog_experiments import (
    LOG_DIR,
    MAPILLARY_ROOT,
    OUTPUT_ROOT,
    PYTHON,
    ROOT,
    append_lab_note,
    append_summary,
    evaluate_variant,
    now,
    run_logged,
    score_metrics,
)


BASELINE_RUN = "synthetic_finetuned_nafnet_20260615"
ABLATION_RUN = "synthetic_finetuned_nafnet_no_pretraining_20260615"
PRESET = ROOT / "presets" / "followup_base_mild_tv.json"
COMPARISON_DIR = OUTPUT_ROOT / "ablation_no_pretraining_20260615"


def train_no_pretraining(variant: dict[str, object]) -> tuple[Path, Path]:
    run_name = str(variant["run_name"])
    run_dir = OUTPUT_ROOT / run_name
    checkpoint = run_dir / "checkpoints" / "best_model.pt"
    config = run_dir / "run_config.json"
    if checkpoint.exists() and config.exists():
        return checkpoint, config

    cmd = [
        str(PYTHON),
        str(ROOT / "train_spatial_mapillary_nafnet.py"),
        "--mapillary-root",
        str(MAPILLARY_ROOT),
        "--preset-json",
        str(variant["preset"]),
        "--out-dir",
        str(OUTPUT_ROOT),
        "--run-name",
        run_name,
        "--epochs",
        "1",
        "--patch-size",
        "512",
        "--batch-size",
        "2",
        "--num-workers",
        "8",
        "--learning-rate",
        str(variant.get("lr", 1e-4)),
        "--seed",
        str(variant.get("seed", 723)),
        "--airlight-jitter",
        str(variant.get("airlight_jitter", 0.03)),
        "--beta-mult-min",
        str(variant.get("beta_min", 0.55)),
        "--beta-mult-max",
        str(variant.get("beta_max", 1.40)),
        "--variation-mult-min",
        str(variant.get("variation_min", 0.60)),
        "--variation-mult-max",
        str(variant.get("variation_max", 1.15)),
        "--light-fog-prob",
        str(variant.get("light_prob", 0.20)),
        "--identity-prob",
        str(variant.get("identity_prob", 0.06)),
        "--loss",
        str(variant.get("loss", "l1")),
        "--color-loss-weight",
        str(variant.get("color_loss_weight", 0.0)),
        "--residual-tv-weight",
        str(variant.get("residual_tv_weight", 0.010)),
        "--max-train-batches",
        str(variant.get("max_train_batches", 2600)),
        "--max-val-batches",
        str(variant.get("max_val_batches", 300)),
        "--max-test-batches",
        str(variant.get("max_test_batches", 500)),
        "--force",
    ]
    run_logged(cmd, LOG_DIR / f"{run_name}_train.log")
    return checkpoint, config


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def first_history(summary: dict[str, object]) -> dict[str, object]:
    history = summary.get("history", [])
    if isinstance(history, list) and history:
        row = history[-1]
        if isinstance(row, dict):
            return row
    return {}


def synthetic_training_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, run_name in [
        ("fog_chamber_pretrained", BASELINE_RUN),
        ("no_pretraining", ABLATION_RUN),
    ]:
        summary_path = OUTPUT_ROOT / run_name / "summary.json"
        summary = read_json(summary_path)
        config = summary.get("run_config", {})
        history = first_history(summary)
        test_metrics = summary.get("test_metrics", {})
        rows.append(
            {
                "model": label,
                "run_name": run_name,
                "init_checkpoint": config.get("init_checkpoint", "") if isinstance(config, dict) else "",
                "train_loss": history.get("train_loss", ""),
                "val_loss": history.get("val_loss", ""),
                "val_psnr": history.get("val_psnr", ""),
                "test_loss": test_metrics.get("test_loss", "") if isinstance(test_metrics, dict) else "",
                "test_psnr": test_metrics.get("test_psnr", "") if isinstance(test_metrics, dict) else "",
                "test_batches": test_metrics.get("test_batches", "") if isinstance(test_metrics, dict) else "",
                "elapsed_seconds": summary.get("elapsed_seconds", ""),
            }
        )
    return rows


def public_metric_comparison_rows() -> list[dict[str, object]]:
    baseline_rows = {
        (row["dataset"], row["split"]): row
        for row in read_csv_rows(OUTPUT_ROOT / f"{BASELINE_RUN}_public_paired_eval" / "paired_public_summary.csv")
    }
    ablation_rows = {
        (row["dataset"], row["split"]): row
        for row in read_csv_rows(OUTPUT_ROOT / f"{ABLATION_RUN}_public_paired_eval" / "paired_public_summary.csv")
    }
    metrics = [
        "pred_psnr",
        "pred_ssim",
        "delta_psnr",
        "delta_ssim",
        "pred_delta_e00",
        "pred_bright_delta_e00",
        "pred_mean_r_bias",
        "pred_mean_g_bias",
        "pred_mean_b_bias",
    ]
    rows: list[dict[str, object]] = []
    for key in sorted(set(baseline_rows) & set(ablation_rows)):
        base = baseline_rows[key]
        abl = ablation_rows[key]
        row: dict[str, object] = {"dataset": key[0], "split": key[1], "n": abl.get("n", "")}
        for metric in metrics:
            base_value = float(base[metric])
            abl_value = float(abl[metric])
            row[f"{metric}_fog_chamber_pretrained"] = base_value
            row[f"{metric}_no_pretraining"] = abl_value
            row[f"{metric}_no_pretraining_minus_pretrained"] = abl_value - base_value
        rows.append(row)
    return rows


def score_rows() -> list[dict[str, object]]:
    rows = []
    for label, run_name in [
        ("fog_chamber_pretrained", BASELINE_RUN),
        ("no_pretraining", ABLATION_RUN),
    ]:
        summary_csv = OUTPUT_ROOT / f"{run_name}_public_paired_eval" / "paired_public_summary.csv"
        rows.append({"model": label, "run_name": run_name, **score_metrics(summary_csv)})
    if len(rows) == 2:
        delta = {"model": "no_pretraining_minus_pretrained", "run_name": ""}
        for key in rows[0]:
            if key not in {"model", "run_name"}:
                delta[key] = float(rows[1][key]) - float(rows[0][key])
        rows.append(delta)
    return rows


def draw_labeled_comparison(baseline_path: Path, ablation_path: Path, out_path: Path, title: str) -> None:
    baseline = Image.open(baseline_path).convert("RGB")
    ablation = Image.open(ablation_path).convert("RGB")
    width = min(baseline.width, ablation.width, 1800)
    baseline = baseline.resize((width, round(baseline.height * width / baseline.width)), Image.Resampling.LANCZOS)
    ablation = ablation.resize((width, round(ablation.height * width / ablation.width)), Image.Resampling.LANCZOS)
    pad = 18
    label_h = 58
    canvas = Image.new("RGB", (width, baseline.height + ablation.height + label_h * 3 + pad * 4), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    y = pad
    draw.text((pad, y), title, fill=(0, 0, 0), font=font)
    y += label_h
    draw.text((pad, y), "Fog-chamber pretrained initialization", fill=(0, 0, 0), font=font)
    y += label_h
    canvas.paste(baseline, (0, y))
    y += baseline.height + pad
    draw.text((pad, y), "No pretraining", fill=(0, 0, 0), font=font)
    y += label_h
    canvas.paste(ablation, (0, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def make_qualitative_comparisons() -> list[Path]:
    baseline_dir = OUTPUT_ROOT / f"{BASELINE_RUN}_review_sheets"
    ablation_dir = OUTPUT_ROOT / f"{ABLATION_RUN}_review_sheets"
    outputs: list[Path] = []
    for filename, title in [
        ("all5_dataset_overview.jpg", "Synthetic fine-tuning ablation overview"),
        ("nh_haze_options.jpg", "NH-HAZE qualitative comparison"),
        ("o_haze_options.jpg", "O-HAZE qualitative comparison"),
        ("ntire26_nh_options.jpg", "NTIRE qualitative comparison"),
        ("airplane_all_options.jpg", "Aircraft-window qualitative comparison"),
        ("fog_machine_options.jpg", "Fog-machine qualitative comparison"),
        ("mapillary_synthetic_val_test_options.jpg", "Synthetic validation/test comparison"),
    ]:
        baseline_path = baseline_dir / filename
        ablation_path = ablation_dir / filename
        if baseline_path.exists() and ablation_path.exists():
            out_path = COMPARISON_DIR / f"pretrained_vs_no_pretraining_{filename}"
            draw_labeled_comparison(baseline_path, ablation_path, out_path, title)
            outputs.append(out_path)
    return outputs


def write_report(qualitative_paths: list[Path]) -> None:
    score_path = COMPARISON_DIR / "score_comparison.csv"
    public_path = COMPARISON_DIR / "public_paired_metric_comparison.csv"
    synthetic_path = COMPARISON_DIR / "synthetic_training_metric_comparison.csv"
    lines = [
        "# Fog-chamber Pretraining Ablation",
        "",
        f"Generated: {now()}",
        "",
        "This compares the selected synthetic fine-tuning run against the same training recipe started from random NAFNet weights.",
        "",
        "## Quantitative Outputs",
        f"- Synthetic training metrics: `{synthetic_path}`",
        f"- Public paired metrics: `{public_path}`",
        f"- Aggregate score metrics: `{score_path}`",
        "",
        "## Qualitative Outputs",
    ]
    lines.extend(f"- `{path}`" for path in qualitative_paths)
    lines.extend(
        [
            "",
            "## Source Runs",
            f"- Fog-chamber pretrained: `{OUTPUT_ROOT / BASELINE_RUN}`",
            f"- No pretraining: `{OUTPUT_ROOT / ABLATION_RUN}`",
        ]
    )
    (COMPARISON_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_outputs() -> None:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(COMPARISON_DIR / "synthetic_training_metric_comparison.csv", synthetic_training_rows())
    write_csv(COMPARISON_DIR / "public_paired_metric_comparison.csv", public_metric_comparison_rows())
    write_csv(COMPARISON_DIR / "score_comparison.csv", score_rows())
    qualitative_paths = make_qualitative_comparisons()
    write_report(qualitative_paths)


def main() -> None:
    if not PRESET.exists():
        raise FileNotFoundError(PRESET)
    variant: dict[str, object] = {
        "run_name": ABLATION_RUN,
        "public_max_per_dataset": 10,
        "preset": PRESET,
        "seed": 723,
        "beta_min": 0.55,
        "beta_max": 1.40,
        "variation_min": 0.60,
        "variation_max": 1.15,
        "light_prob": 0.20,
        "identity_prob": 0.06,
        "color_loss_weight": 0.0,
        "residual_tv_weight": 0.010,
        "max_train_batches": 2600,
        "max_val_batches": 300,
        "max_test_batches": 500,
    }

    append_lab_note(
        "# No-pretraining ablation\n"
        f"Started {now()}\n"
        f"Baseline: `{BASELINE_RUN}`\n"
        f"Ablation: `{ABLATION_RUN}`\n"
        "Only initialization changes: the ablation omits the fog-chamber checkpoint and starts from random NAFNet weights."
    )
    started = time.time()
    checkpoint, config = train_no_pretraining(variant)
    paired_dir, airplane_dir, free_dir, sheets_dir = evaluate_variant(ABLATION_RUN, checkpoint, config, variant)
    stats = score_metrics(paired_dir / "paired_public_summary.csv")
    append_summary(
        {
            "run_name": ABLATION_RUN,
            "finished": now(),
            "checkpoint": str(checkpoint),
            "paired_summary": str(paired_dir / "paired_public_summary.csv"),
            "review_sheet": str(sheets_dir / "all5_dataset_overview.jpg"),
            "airplane_dir": str(airplane_dir),
            "free_fog_dir": str(free_dir),
            **stats,
        }
    )
    write_comparison_outputs()
    append_lab_note(
        f"Finished no-pretraining ablation {now()} in {(time.time() - started) / 60.0:.1f} min.\n"
        f"Comparison report: `{COMPARISON_DIR / 'README.md'}`\n"
        f"Score: {stats['score']:.3f}"
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        append_lab_note(f"FAILED no-pretraining ablation {now()}: {exc!r}\n")
        raise
