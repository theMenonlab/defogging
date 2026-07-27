#!/usr/bin/env python3
"""Shared helpers for the paper synthetic fine-tuning runner."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = Path(os.environ.get("PYTHON", "python"))
MAPILLARY_ROOT = Path(os.environ.get("MAPILLARY_ROOT", "data/mapillary_vistas"))
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "outputs/synthetic_finetuned_nafnet"))
AIRPLANE_ALL = Path(os.environ.get("AIRCRAFT_INPUT_DIR", str(OUTPUT_ROOT / "test_inputs" / "aircraft")))
FREE_FOG_INPUTS = Path(os.environ.get("FREE_FOG_INPUT_DIR", "data/free_fog"))
INIT_CHECKPOINT = Path(os.environ.get("INIT_CHECKPOINT", "models/fog_chamber_nafnet_model_state_20260615.pth"))
INFER_SCRIPT = Path(os.environ.get("INFER_SCRIPT", str(ROOT.parent / "nafnet_finetuning" / "infer_nafnet_fog.py")))
BASE_PRESET = ROOT / "spatial_fog_preset.json"
LOG_DIR = ROOT / "logs"
SUMMARY_CSV = OUTPUT_ROOT / "overnight_20260613_experiment_summary.csv"
NOTEBOOK = OUTPUT_ROOT / "overnight_20260613_lab_notes.md"


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_logged(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["TQDM_MININTERVAL"] = "30"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# {now()}\n")
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        process = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
        rc = process.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def make_preset(name: str, updates: dict[str, float]) -> Path:
    with BASE_PRESET.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    params = deepcopy(payload["params"])
    params.update(updates)
    params["name"] = name
    out = ROOT / "presets" / f"{name}.json"
    write_json(out, {"model_type": payload.get("model_type", "spatial_gaussian_field_v1"), "description": f"Overnight variant {name}", "params": params})
    return out


def read_summary(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {f"{row['dataset']}:{row['split']}": row for row in csv.DictReader(handle)}


def safe_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def score_metrics(summary_csv: Path) -> dict[str, float]:
    rows = read_summary(summary_csv)
    all_rows = [rows[key] for key in ["NH-HAZE:all", "O-HAZE:all", "NTIRE26-NH:all"] if key in rows]
    test_rows = [rows[key] for key in ["NH-HAZE:test", "O-HAZE:test", "NTIRE26-NH:test"] if key in rows]
    use_rows = all_rows if all_rows else test_rows
    mean_delta_psnr = sum(safe_float(row, "delta_psnr") for row in use_rows) / max(1, len(use_rows))
    mean_delta_ssim = sum(safe_float(row, "delta_ssim") for row in use_rows) / max(1, len(use_rows))
    mean_pred_de = sum(safe_float(row, "pred_delta_e00") for row in use_rows) / max(1, len(use_rows))
    mean_bright_de = sum(safe_float(row, "pred_bright_delta_e00") for row in use_rows) / max(1, len(use_rows))
    rgb_bias = sum(
        abs(safe_float(row, "pred_mean_r_bias")) + abs(safe_float(row, "pred_mean_g_bias")) + abs(safe_float(row, "pred_mean_b_bias"))
        for row in use_rows
    ) / max(1, len(use_rows))
    score = mean_delta_psnr + 4.0 * mean_delta_ssim - 0.035 * mean_pred_de - 0.020 * mean_bright_de - 1.5 * rgb_bias
    return {
        "mean_delta_psnr": mean_delta_psnr,
        "mean_delta_ssim": mean_delta_ssim,
        "mean_pred_delta_e00": mean_pred_de,
        "mean_pred_bright_delta_e00": mean_bright_de,
        "mean_abs_rgb_bias_sum": rgb_bias,
        "score": score,
    }


def append_lab_note(text: str) -> None:
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    with NOTEBOOK.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n\n")


def append_summary(row: dict[str, object]) -> None:
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    exists = SUMMARY_CSV.exists()
    with SUMMARY_CSV.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train_variant(variant: dict[str, object]) -> tuple[Path, Path]:
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
        "--init-checkpoint",
        str(INIT_CHECKPOINT),
        "--strict-load",
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
        str(variant.get("seed", 700)),
        "--airlight-jitter",
        str(variant.get("airlight_jitter", 0.03)),
        "--beta-mult-min",
        str(variant.get("beta_min", 0.5)),
        "--beta-mult-max",
        str(variant.get("beta_max", 1.45)),
        "--variation-mult-min",
        str(variant.get("variation_min", 0.6)),
        "--variation-mult-max",
        str(variant.get("variation_max", 1.25)),
        "--light-fog-prob",
        str(variant.get("light_prob", 0.18)),
        "--identity-prob",
        str(variant.get("identity_prob", 0.04)),
        "--loss",
        str(variant.get("loss", "l1")),
        "--color-loss-weight",
        str(variant.get("color_loss_weight", 0.10)),
        "--residual-tv-weight",
        str(variant.get("residual_tv_weight", 0.015)),
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


def evaluate_variant(run_name: str, checkpoint: Path, config: Path, variant: dict[str, object]) -> tuple[Path, Path, Path, Path]:
    paired_dir = OUTPUT_ROOT / f"{run_name}_public_paired_eval"
    airplane_dir = OUTPUT_ROOT / f"{run_name}_airplane_all"
    free_dir = OUTPUT_ROOT / f"{run_name}_free_fog"
    sheets_dir = OUTPUT_ROOT / f"{run_name}_review_sheets"
    if not (paired_dir / "paired_public_summary.csv").exists():
        public_cmd = [
                str(PYTHON),
                str(ROOT / "evaluate_public_paired_checkpoint.py"),
                "--checkpoint",
                str(checkpoint),
                "--run-config",
                str(config),
                "--output-dir",
                str(paired_dir),
                "--tile-size",
                "1024",
                "--tile-overlap",
                "96",
        ]
        if variant.get("public_max_per_dataset") is not None:
            public_cmd.extend(["--max-records-per-dataset", str(variant["public_max_per_dataset"])])
        run_logged(
            public_cmd,
            LOG_DIR / f"{run_name}_public_eval.log",
        )
    if not (airplane_dir / "inference_manifest.json").exists():
        run_logged(
            [
                str(PYTHON),
                str(INFER_SCRIPT),
                "--input-dir",
                str(AIRPLANE_ALL),
                "--output-dir",
                str(airplane_dir),
                "--checkpoint",
                str(checkpoint),
                "--run-config",
                str(config),
                "--tile-size",
                "1024",
                "--tile-overlap",
                "96",
                "--save-side-by-side",
                "--overwrite",
            ],
            LOG_DIR / f"{run_name}_airplane_all_infer.log",
        )
    if not (free_dir / "inference_manifest.json").exists():
        run_logged(
            [
                str(PYTHON),
                str(INFER_SCRIPT),
                "--input-dir",
                str(FREE_FOG_INPUTS),
                "--output-dir",
                str(free_dir),
                "--checkpoint",
                str(checkpoint),
                "--run-config",
                str(config),
                "--tile-size",
                "1024",
                "--tile-overlap",
                "96",
                "--save-side-by-side",
                "--overwrite",
            ],
            LOG_DIR / f"{run_name}_free_fog_infer.log",
        )
    if not (sheets_dir / "all5_dataset_overview.jpg").exists():
        run_logged(
            [
                str(PYTHON),
                str(ROOT / "make_experiment_review_sheets.py"),
                "--run-name",
                run_name,
                "--run-dir",
                str(OUTPUT_ROOT / run_name),
                "--paired-dir",
                str(paired_dir),
                "--airplane-dir",
                str(airplane_dir),
                "--free-fog-dir",
                str(free_dir),
                "--sheet-dir",
                str(sheets_dir),
            ],
            LOG_DIR / f"{run_name}_review_sheets.log",
        )
    return paired_dir, airplane_dir, free_dir, sheets_dir


def main() -> None:
    raise SystemExit(
        "This module provides shared helpers. Run "
        "run_followup_synthetic_fog_experiments.py for the paper synthetic fine-tuning configuration."
    )


if __name__ == "__main__":
    main()
