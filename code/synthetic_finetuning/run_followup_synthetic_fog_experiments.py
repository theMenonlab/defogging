#!/usr/bin/env python3
"""Run the paper synthetic fine-tuning configuration."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from run_overnight_synthetic_fog_experiments import (
    BASE_PRESET,
    OUTPUT_ROOT,
    append_lab_note,
    append_summary,
    evaluate_variant,
    make_preset,
    now,
    score_metrics,
    train_variant,
)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_pid(pid: int | None) -> None:
    if not pid:
        return
    append_lab_note(f"# Follow-up synthetic fog experiments\nWaiting for PID {pid} at {now()}\n")
    while pid_alive(pid):
        time.sleep(300)


def run_variant(variant: dict[str, object]) -> None:
    run_name = str(variant["run_name"])
    append_lab_note(f"## {run_name}\nStarted {now()}\nPreset: `{variant['preset']}`\n")
    checkpoint, config = train_variant(variant)
    paired_dir, airplane_dir, free_dir, sheets_dir = evaluate_variant(run_name, checkpoint, config, variant)
    summary_csv = paired_dir / "paired_public_summary.csv"
    stats = score_metrics(summary_csv)
    row = {
        "run_name": run_name,
        "finished": now(),
        "checkpoint": str(checkpoint),
        "paired_summary": str(summary_csv),
        "review_sheet": str(sheets_dir / "all5_dataset_overview.jpg"),
        "airplane_dir": str(airplane_dir),
        "free_fog_dir": str(free_dir),
        **stats,
    }
    append_summary(row)
    append_lab_note(
        f"Finished {now()}.\n"
        f"- mean Delta PSNR: {stats['mean_delta_psnr']:.3f}\n"
        f"- mean Delta SSIM: {stats['mean_delta_ssim']:.4f}\n"
        f"- mean pred DeltaE00: {stats['mean_pred_delta_e00']:.3f}\n"
        f"- mean bright pred DeltaE00: {stats['mean_pred_bright_delta_e00']:.3f}\n"
        f"- mean abs RGB bias sum: {stats['mean_abs_rgb_bias_sum']:.4f}\n"
        f"- score: {stats['score']:.3f}\n"
        f"Review sheet: `{sheets_dir / 'all5_dataset_overview.jpg'}`\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, default=None)
    args = parser.parse_args()

    wait_for_pid(args.wait_pid)
    append_lab_note(f"# Follow-up synthetic fog experiments\nStarted {now()}\n")

    mild_tv_preset = make_preset(
        "followup_base_mild_tv",
        {"bloom_strength": 0.28, "bloom_radius": 12.0, "edge_veil_strength": 0.18, "saturation_mix": 0.22},
    )

    variants: list[dict[str, object]] = [
        {
            "run_name": "synthetic_finetuned_nafnet_20260615",
            "public_max_per_dataset": 10,
            "preset": mild_tv_preset,
            "seed": 723,
            "beta_min": 0.55,
            "beta_max": 1.40,
            "variation_min": 0.60,
            "variation_max": 1.15,
            "light_prob": 0.20,
            "identity_prob": 0.06,
            "color_loss_weight": 0.0,
            "residual_tv_weight": 0.010,
        },
    ]

    for variant in variants:
        run_variant(variant)

    append_lab_note(f"Completed follow-up queue {now()}\nSummary CSV: `{OUTPUT_ROOT / 'overnight_20260613_experiment_summary.csv'}`\n")


if __name__ == "__main__":
    main()
