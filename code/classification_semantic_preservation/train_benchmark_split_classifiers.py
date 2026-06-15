#!/usr/bin/env python3
"""Retrain fog-chamber category classifiers with the benchmark held-out split.

The test split is read from fog_chamber_nafnet/metrics.csv, which is the same
552-image held-out set used by the NAFNet fog-chamber benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import classification_train_resnet50_base as base  # noqa: E402


ROOT = Path(os.environ.get("DEFOG_WORK_ROOT", "."))
DEFAULT_BENCHMARK = ROOT / "fog_chamber_nafnet/metrics.csv"
DEFAULT_FOG_ROOT = Path(os.environ.get("FOG_CHAMBER_FOG_ROOT", "data/VerticalFilter_MediumFog_Redo_3-21-26_aligned"))
DEFAULT_GT_ROOT = Path(os.environ.get("FOG_CHAMBER_GT_ROOT", "data/archive_gt_matched"))
DEFAULT_OUTPUT = ROOT / "outputs/fog_chamber_classification_benchmark_20260604"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-metrics", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--fog-root", type=Path, default=DEFAULT_FOG_ROOT)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["clear_targets", "foggy_inputs"],
        choices=["clear_targets", "foggy_inputs"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--model-name", default="resnet50", choices=sorted(base.MODEL_REGISTRY.keys()))
    parser.add_argument("--init-weights", default="imagenet", choices=["imagenet", "scratch"])
    parser.add_argument("--augment-preset", default="mild", choices=["mild", "standard", "strong"])
    parser.add_argument("--head-type", default="linear", choices=["linear", "mlp"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--mlp-hidden-ratio", type=float, default=0.5)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--scheduler", default="cosine", choices=["cosine", "none"])
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--class-weighting", default="none", choices=["none", "balanced"])
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--cutmix-alpha", type=float, default=0.0)
    parser.add_argument("--mixup-prob", type=float, default=0.0)
    parser.add_argument("--mixup-switch-prob", type=float, default=0.5)
    parser.add_argument("--random-erasing-prob", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true", help="Build manifests and exit before training.")
    return parser.parse_args()


def image_files(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_benchmark_ids(benchmark_metrics: Path) -> tuple[pd.DataFrame, set[tuple[str, str]]]:
    df = pd.read_csv(benchmark_metrics)
    required = {"category", "image_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Benchmark metrics missing columns: {sorted(missing)}")
    ids = set(zip(df["category"].astype(str), df["image_name"].astype(str)))
    if len(ids) != len(df):
        raise ValueError("Benchmark metrics contains duplicate category/image_name rows")
    return df, ids


def collect_paired_samples(
    fog_root: Path, gt_root: Path, benchmark_ids: set[tuple[str, str]]
) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, str]] = []
    categories = sorted(p.name for p in fog_root.iterdir() if p.is_dir())
    for category in categories:
        fog_dir = fog_root / category
        gt_dir = gt_root / category
        if not gt_dir.is_dir():
            continue
        gt_by_name = {p.name: p for p in image_files(gt_dir)}
        for fog_path in image_files(fog_dir):
            gt_path = gt_by_name.get(fog_path.name)
            if gt_path is None:
                continue
            rows.append(
                {
                    "category": category,
                    "image_name": fog_path.name,
                    "pair_id": f"{category}/{fog_path.name}",
                    "fog_path": str(fog_path.resolve()),
                    "gt_path": str(gt_path.resolve()),
                    "benchmark_split": "test"
                    if (category, fog_path.name) in benchmark_ids
                    else "trainval",
                }
            )
    df = pd.DataFrame(rows)
    missing_benchmark = sorted(benchmark_ids - set(zip(df["category"], df["image_name"])))
    if missing_benchmark:
        preview = ", ".join(f"{c}/{n}" for c, n in missing_benchmark[:10])
        raise FileNotFoundError(f"Missing {len(missing_benchmark)} benchmark images: {preview}")
    class_names = sorted(df["category"].unique().tolist())
    if len(class_names) != 6:
        raise ValueError(f"Expected 6 benchmark classes, found {class_names}")
    return df, class_names


def build_condition_manifest(
    paired_df: pd.DataFrame,
    class_names: list[str],
    condition: str,
    seed: int,
    val_fraction: float,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, int], dict[str, str]]:
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {str(v): k for k, v in class_to_idx.items()}
    path_column = "gt_path" if condition == "clear_targets" else "fog_path"

    trainval = paired_df[paired_df["benchmark_split"] == "trainval"].copy()
    test = paired_df[paired_df["benchmark_split"] == "test"].copy()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(trainval, trainval["category"]))
    trainval["split"] = "train"
    trainval.iloc[val_idx, trainval.columns.get_loc("split")] = "val"
    test["split"] = "test"
    manifest = pd.concat([trainval.iloc[train_idx], trainval.iloc[val_idx], test], ignore_index=True)
    manifest["condition"] = condition
    manifest["path"] = manifest[path_column]
    manifest["label_name"] = manifest["category"]
    manifest["label_idx"] = manifest["category"].map(class_to_idx).astype(int)
    manifest = manifest.sort_values(["split", "category", "image_name"]).reset_index(drop=True)

    by_split_and_class: dict[str, dict[str, int]] = {}
    for split_name in ["train", "val", "test"]:
        counts = manifest[manifest["split"] == split_name]["category"].value_counts().sort_index()
        by_split_and_class[split_name] = {k: int(v) for k, v in counts.items()}
    split_counts = {
        "total": int(len(manifest)),
        "train": int((manifest["split"] == "train").sum()),
        "val": int((manifest["split"] == "val").sum()),
        "test": int((manifest["split"] == "test").sum()),
        "by_split_and_class": by_split_and_class,
        "test_definition": "same 552 image IDs as fog_chamber_nafnet/metrics.csv",
    }
    return manifest, split_counts, class_to_idx, idx_to_class


def validate_fixed_split(manifest: pd.DataFrame, benchmark_count: int) -> None:
    test = manifest[manifest["split"] == "test"]
    if len(test) != benchmark_count:
        raise RuntimeError(f"Expected {benchmark_count} benchmark test rows, found {len(test)}")
    overlap = set(manifest[manifest["split"] == "test"]["pair_id"]) & set(
        manifest[manifest["split"].isin(["train", "val"])]["pair_id"]
    )
    if overlap:
        raise RuntimeError(f"Benchmark test leakage into train/val: {sorted(overlap)[:5]}")
    counts = test["category"].value_counts().to_dict()
    if sorted(counts.values()) != [92, 92, 92, 92, 92, 92]:
        raise RuntimeError(f"Unexpected benchmark test class counts: {counts}")


def train_condition(
    condition: str,
    manifest_df: pd.DataFrame,
    split_counts: dict[str, Any],
    class_to_idx: dict[str, int],
    idx_to_class: dict[str, str],
    args: argparse.Namespace,
) -> Path:
    base.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    amp_enabled = use_cuda

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    init_tag = "pretrained" if args.init_weights == "imagenet" else "scratch"
    run_dir = args.output_root / condition / f"{timestamp}_{args.model_name}_{init_tag}_benchmark_split"
    run_dir.mkdir(parents=True, exist_ok=False)

    env_summary = {
        "python": sys.version.replace("\n", " "),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (run_dir / "env_summary.json").write_text(json.dumps(env_summary, indent=2), encoding="utf-8")

    run_config = vars(args).copy()
    run_config["condition"] = condition
    run_config["fixed_test_split"] = split_counts["test_definition"]
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")
    manifest_df.to_csv(run_dir / "split_manifest.csv", index=False)
    (run_dir / "class_to_idx.json").write_text(json.dumps(class_to_idx, indent=2), encoding="utf-8")
    (run_dir / "idx_to_class.json").write_text(json.dumps(idx_to_class, indent=2), encoding="utf-8")

    dataloaders = base.build_dataloaders(
        manifest=manifest_df,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        seed=args.seed,
        use_cuda=use_cuda,
        augment_preset=args.augment_preset,
        random_erasing_prob=args.random_erasing_prob,
    )
    model = base.build_model(
        model_name=args.model_name,
        init_weights=args.init_weights,
        num_classes=len(class_to_idx),
        head_type=args.head_type,
        dropout=args.dropout,
        mlp_hidden_ratio=args.mlp_hidden_ratio,
    ).to(device)

    if args.class_weighting == "balanced":
        train_counts = manifest_df[manifest_df["split"] == "train"]["label_idx"].value_counts().sort_index()
        counts = train_counts.values.astype(np.float32)
        weights = float(counts.sum()) / (len(counts) * counts)
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    else:
        class_weights = None
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, weight=class_weights)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    history_rows: list[dict[str, Any]] = []
    history_path = run_dir / "training_history.csv"
    best_model_path = run_dir / "best_model.pt"
    last_model_path = run_dir / "last_model.pt"
    best_val_macro_f1 = -1.0
    best_epoch = 0
    best_stage = ""
    epoch_counter = 0
    epochs_without_improve = 0

    if args.head_epochs > 0:
        base.freeze_backbone(model)
        optimizer = base.make_optimizer(
            args.optimizer,
            [p for p in model.parameters() if p.requires_grad],
            args.head_lr,
            args.weight_decay,
            args.momentum,
        )
        for stage_epoch in range(1, args.head_epochs + 1):
            epoch_counter += 1
            started_at = time.time()
            train_stats = base.train_one_epoch(
                model,
                dataloaders["train"],
                optimizer,
                criterion,
                device,
                scaler,
                amp_enabled,
                args.mixup_alpha,
                args.cutmix_alpha,
                args.mixup_prob,
                args.mixup_switch_prob,
            )
            val_stats = base.evaluate(model, dataloaders["val"], criterion, device, amp_enabled)
            row = {
                "epoch": epoch_counter,
                "stage_epoch": stage_epoch,
                "stage": "head",
                "train_loss": train_stats["loss"],
                "train_accuracy": train_stats["accuracy"],
                "val_loss": val_stats["loss"],
                "val_accuracy": val_stats["accuracy"],
                "val_macro_precision": val_stats["macro_precision"],
                "val_macro_recall": val_stats["macro_recall"],
                "val_macro_f1": val_stats["macro_f1"],
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_seconds": time.time() - started_at,
            }
            history_rows.append(row)
            pd.DataFrame(history_rows).to_csv(history_path, index=False)
            print(
                f"[{condition}][Epoch {epoch_counter:02d}][head] "
                f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
                f"val_macro_f1={row['val_macro_f1']:.4f}",
                flush=True,
            )
            if val_stats["macro_f1"] > (best_val_macro_f1 + args.min_delta):
                best_val_macro_f1 = float(val_stats["macro_f1"])
                best_epoch = epoch_counter
                best_stage = "head"
                base.save_checkpoint(
                    best_model_path,
                    model,
                    epoch_counter,
                    best_stage,
                    best_val_macro_f1,
                    class_to_idx,
                    idx_to_class,
                    args,
                )

    if args.finetune_epochs > 0:
        base.unfreeze_all(model)
        optimizer = base.make_optimizer(
            args.optimizer, model.parameters(), args.finetune_lr, args.weight_decay, args.momentum
        )
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs)
            if args.scheduler == "cosine"
            else None
        )
        for stage_epoch in range(1, args.finetune_epochs + 1):
            epoch_counter += 1
            started_at = time.time()
            train_stats = base.train_one_epoch(
                model,
                dataloaders["train"],
                optimizer,
                criterion,
                device,
                scaler,
                amp_enabled,
                args.mixup_alpha,
                args.cutmix_alpha,
                args.mixup_prob,
                args.mixup_switch_prob,
            )
            val_stats = base.evaluate(model, dataloaders["val"], criterion, device, amp_enabled)
            if scheduler is not None:
                scheduler.step()
            row = {
                "epoch": epoch_counter,
                "stage_epoch": stage_epoch,
                "stage": "finetune",
                "train_loss": train_stats["loss"],
                "train_accuracy": train_stats["accuracy"],
                "val_loss": val_stats["loss"],
                "val_accuracy": val_stats["accuracy"],
                "val_macro_precision": val_stats["macro_precision"],
                "val_macro_recall": val_stats["macro_recall"],
                "val_macro_f1": val_stats["macro_f1"],
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_seconds": time.time() - started_at,
            }
            history_rows.append(row)
            pd.DataFrame(history_rows).to_csv(history_path, index=False)
            print(
                f"[{condition}][Epoch {epoch_counter:02d}][finetune] "
                f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
                f"val_macro_f1={row['val_macro_f1']:.4f}",
                flush=True,
            )
            if val_stats["macro_f1"] > (best_val_macro_f1 + args.min_delta):
                best_val_macro_f1 = float(val_stats["macro_f1"])
                best_epoch = epoch_counter
                best_stage = "finetune"
                epochs_without_improve = 0
                base.save_checkpoint(
                    best_model_path,
                    model,
                    epoch_counter,
                    best_stage,
                    best_val_macro_f1,
                    class_to_idx,
                    idx_to_class,
                    args,
                )
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= args.patience:
                    print(f"[{condition}] Early stopping after {epochs_without_improve} epochs.", flush=True)
                    break

    base.save_checkpoint(
        last_model_path,
        model,
        epoch_counter,
        "last",
        best_val_macro_f1,
        class_to_idx,
        idx_to_class,
        args,
    )
    if not best_model_path.exists():
        base.save_checkpoint(
            best_model_path,
            model,
            epoch_counter,
            "last",
            best_val_macro_f1,
            class_to_idx,
            idx_to_class,
            args,
        )
        best_epoch = epoch_counter
        best_stage = "last"

    best_checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_stats = base.evaluate(model, dataloaders["test"], criterion, device, amp_enabled)

    y_true = test_stats["y_true"]
    y_pred = test_stats["y_pred"]
    y_probs = test_stats["y_probs"]
    paths = test_stats["paths"]
    confidences = y_probs.max(axis=1)
    idx_to_label = {idx: name for name, idx in class_to_idx.items()}
    pred_labels = [idx_to_label[int(i)] for i in y_pred]
    true_labels = [idx_to_label[int(i)] for i in y_true]
    correct = (y_true == y_pred).astype(int)
    predictions_df = pd.DataFrame(
        {
            "path": paths,
            "true_label": true_labels,
            "pred_label": pred_labels,
            "confidence": confidences,
            "correct": correct,
        }
    )
    predictions_df.to_csv(run_dir / "test_predictions.csv", index=False)
    predictions_df[predictions_df["correct"] == 0].to_csv(run_dir / "misclassified_samples.csv", index=False)

    report_dict = classification_report(
        y_true, y_pred, target_names=list(class_to_idx.keys()), output_dict=True, zero_division=0
    )
    per_class_rows = []
    for class_name in class_to_idx:
        row = report_dict[class_name]
        per_class_rows.append(
            {
                "class_name": class_name,
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1_score": float(row["f1-score"]),
                "support": int(row["support"]),
            }
        )
    pd.DataFrame(per_class_rows).to_csv(run_dir / "classification_report.csv", index=False)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_to_idx))))
    base.save_confusion_outputs(
        cm,
        list(class_to_idx.keys()),
        run_dir / "confusion_matrix.csv",
        run_dir / "confusion_matrix.png",
    )
    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(history_path, index=False)
    base.save_training_curves(history_df, run_dir / "training_curves.png")

    metrics = {
        "run_id": run_dir.name,
        "condition": condition,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": args.model_name,
        "init_weights": args.init_weights,
        "augment_preset": args.augment_preset,
        "head_type": args.head_type,
        "optimizer": args.optimizer,
        "class_names": list(class_to_idx.keys()),
        "split_counts": split_counts,
        "best_epoch": int(best_epoch),
        "best_stage": best_stage,
        "val_best_macro_f1": float(best_val_macro_f1),
        "test_definition": split_counts["test_definition"],
        "test_loss": float(test_stats["loss"]),
        "test_accuracy": float(test_stats["accuracy"]),
        "test_top2_accuracy": float(test_stats["top2_accuracy"]),
        "test_macro_precision": float(test_stats["macro_precision"]),
        "test_macro_recall": float(test_stats["macro_recall"]),
        "test_macro_f1": float(test_stats["macro_f1"]),
        "test_micro_f1": float(test_stats["micro_f1"]),
        "test_micro_precision": float(test_stats["micro_precision"]),
        "test_micro_recall": float(test_stats["micro_recall"]),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"[{condition}] complete: acc={metrics['test_accuracy']:.4f}, "
        f"macro_F1={metrics['test_macro_f1']:.4f}, top2={metrics['test_top2_accuracy']:.4f}",
        flush=True,
    )
    return run_dir


def main() -> None:
    args = parse_args()
    if args.val_fraction <= 0 or args.val_fraction >= 1:
        raise ValueError("--val-fraction must be in (0, 1)")
    args.output_root.mkdir(parents=True, exist_ok=True)
    benchmark_df, benchmark_ids = load_benchmark_ids(args.benchmark_metrics)
    paired_df, class_names = collect_paired_samples(args.fog_root, args.gt_root, benchmark_ids)
    paired_df.to_csv(args.output_root / "paired_benchmark_split_source_manifest.csv", index=False)

    run_dirs: dict[str, str] = {}
    for condition in args.conditions:
        manifest, split_counts, class_to_idx, idx_to_class = build_condition_manifest(
            paired_df, class_names, condition, args.seed, args.val_fraction
        )
        validate_fixed_split(manifest, len(benchmark_df))
        condition_dir = args.output_root / condition
        condition_dir.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(condition_dir / "benchmark_fixed_split_manifest.csv", index=False)
        (condition_dir / "split_counts.json").write_text(json.dumps(split_counts, indent=2), encoding="utf-8")
        if args.dry_run:
            run_dirs[condition] = "dry_run_no_training"
        else:
            run_dirs[condition] = str(
                train_condition(condition, manifest, split_counts, class_to_idx, idx_to_class, args)
            )

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "benchmark_metrics": str(args.benchmark_metrics),
        "fog_root": str(args.fog_root),
        "gt_root": str(args.gt_root),
        "benchmark_rows": int(len(benchmark_df)),
        "class_counts_test": benchmark_df["category"].value_counts().sort_index().astype(int).to_dict(),
        "paired_source_rows": int(len(paired_df)),
        "run_dirs": run_dirs,
    }
    (args.output_root / "classification_benchmark_redo_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
