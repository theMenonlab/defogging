#!/usr/bin/env python3
"""Five-class clear-trained recognition check on clear, foggy, and restored images."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from skimage.metrics import structural_similarity
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import classification_train_resnet50_base as base  # noqa: E402
import evaluate_restoration_classification_loop as restore_eval  # noqa: E402
import train_benchmark_split_classifiers as trainer  # noqa: E402


ROOT = Path(os.environ.get("DEFOG_WORK_ROOT", "."))
DEFAULT_OUTPUT = SCRIPT_DIR / "results"
DEFAULT_EXCLUDE = ["illustrations"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-metrics", type=Path, default=ROOT / "fog_chamber_nafnet/metrics.csv")
    parser.add_argument("--fog-root", type=Path, default=trainer.DEFAULT_FOG_ROOT)
    parser.add_argument("--gt-root", type=Path, default=trainer.DEFAULT_GT_ROOT)
    parser.add_argument(
        "--nafnet-checkpoint",
        type=Path,
        default=ROOT / "fog_chamber_nafnet/nafnet_fc_clean_model_state_20260603.pth",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--exclude-categories", nargs="+", default=DEFAULT_EXCLUDE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
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
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--overwrite-restored", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def filter_categories(df: pd.DataFrame, excluded: set[str]) -> pd.DataFrame:
    return df[~df["category"].astype(str).isin(excluded)].copy().reset_index(drop=True)


def validate_five_class_split(manifest: pd.DataFrame, expected_test_count: int) -> None:
    test = manifest[manifest["split"] == "test"]
    if len(test) != expected_test_count:
        raise RuntimeError(f"Expected {expected_test_count} test rows, found {len(test)}")
    overlap = set(test["pair_id"]) & set(manifest[manifest["split"].isin(["train", "val"])]["pair_id"])
    if overlap:
        raise RuntimeError(f"Test leakage into train/val: {sorted(overlap)[:5]}")
    counts = test["category"].value_counts().sort_index().astype(int).to_dict()
    if sorted(counts.values()) != [92] * len(counts):
        raise RuntimeError(f"Unexpected test class counts: {counts}")


def build_restored_manifest(foggy_manifest: pd.DataFrame, restored_root: Path) -> pd.DataFrame:
    test_manifest = foggy_manifest[foggy_manifest["split"] == "test"].copy()
    output_paths: list[str] = []
    for row in test_manifest.itertuples(index=False):
        output_path = restored_root / str(row.category) / f"{Path(str(row.image_name)).stem}.png"
        output_paths.append(str(output_path.resolve()))
    test_manifest["foggy_path"] = test_manifest["path"]
    test_manifest["path"] = output_paths
    test_manifest["condition"] = "nafnet_restored_outputs"
    return test_manifest.reset_index(drop=True)


def evaluate_checkpoint_on_manifest(
    eval_name: str,
    checkpoint_path: Path,
    eval_manifest: pd.DataFrame,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    class_to_idx = {str(k): int(v) for k, v in checkpoint["class_to_idx"].items()}
    idx_to_class = {int(v): str(k) for k, v in class_to_idx.items()}

    manifest = eval_manifest.copy()
    manifest["label_idx"] = manifest["label_name"].map(class_to_idx).astype(int)
    manifest["split"] = "test"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    dataset = base.PathImageDataset(
        paths=manifest["path"].tolist(),
        labels=manifest["label_idx"].astype(int).tolist(),
        transform=restore_eval.eval_transform(int(args.get("img_size", 224))),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
    )

    model = base.build_model(
        model_name=args.get("model_name", "resnet50"),
        init_weights="scratch",
        num_classes=len(class_to_idx),
        head_type=args.get("head_type", "linear"),
        dropout=float(args.get("dropout", 0.0)),
        mlp_hidden_ratio=float(args.get("mlp_hidden_ratio", 0.5)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    criterion = nn.CrossEntropyLoss()
    stats = base.evaluate(model, dataloader, criterion, device, amp_enabled=use_cuda)

    y_true = stats["y_true"]
    y_pred = stats["y_pred"]
    y_probs = stats["y_probs"]
    confidences = y_probs.max(axis=1)
    top2_indices = np.argsort(y_probs, axis=1)[:, -2:][:, ::-1] if y_probs.shape[1] >= 2 else y_pred[:, None]
    top2_labels = [";".join(idx_to_class[int(i)] for i in row) for row in top2_indices]
    pred_labels = [idx_to_class[int(i)] for i in y_pred]
    true_labels = [idx_to_class[int(i)] for i in y_true]
    correct = (y_true == y_pred).astype(int)

    condition_dir = output_dir / eval_name
    condition_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(
        {
            "pair_id": manifest["pair_id"].tolist(),
            "path": stats["paths"],
            "true_label": true_labels,
            "pred_label": pred_labels,
            "top2_labels": top2_labels,
            "confidence": confidences,
            "correct": correct,
        }
    )
    if "foggy_path" in manifest.columns:
        predictions["foggy_path"] = manifest["foggy_path"].tolist()
    predictions.to_csv(condition_dir / "test_predictions.csv", index=False)
    predictions[predictions["correct"] == 0].to_csv(condition_dir / "misclassified_samples.csv", index=False)

    target_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    per_class = []
    for class_name in target_names:
        row = report[class_name]
        per_class.append(
            {
                "class_name": class_name,
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1_score": float(row["f1-score"]),
                "support": int(row["support"]),
            }
        )
    pd.DataFrame(per_class).to_csv(condition_dir / "classification_report.csv", index=False)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(target_names))))
    base.save_confusion_outputs(
        cm,
        target_names,
        condition_dir / "confusion_matrix.csv",
        condition_dir / "confusion_matrix.png",
    )

    metrics = {
        "condition": eval_name,
        "checkpoint": str(checkpoint_path),
        "test_count": int(len(manifest)),
        "test_loss": float(stats["loss"]),
        "test_accuracy": float(stats["accuracy"]),
        "test_top2_accuracy": float(stats["top2_accuracy"]),
        "test_macro_precision": float(stats["macro_precision"]),
        "test_macro_recall": float(stats["macro_recall"]),
        "test_macro_f1": float(stats["macro_f1"]),
        "test_micro_precision": float(stats["micro_precision"]),
        "test_micro_recall": float(stats["micro_recall"]),
        "test_micro_f1": float(stats["micro_f1"]),
    }
    (condition_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def compute_restoration_metrics(restored_manifest: pd.DataFrame, output_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for row in restored_manifest.itertuples(index=False):
        pred = np.asarray(Image.open(row.path).convert("RGB"), dtype=np.float32) / 255.0
        gt = np.asarray(Image.open(row.gt_path).convert("RGB"), dtype=np.float32) / 255.0
        mse = float(np.mean((pred - gt) ** 2))
        mae = float(np.mean(np.abs(pred - gt)))
        psnr = 10.0 * math.log10(1.0 / mse) if mse > 0 else float("inf")
        ssim = float(structural_similarity(gt, pred, channel_axis=2, data_range=1.0))
        rows.append(
            {
                "category": row.category,
                "image_name": row.image_name,
                "mae": mae,
                "mse": mse,
                "psnr": psnr,
                "ssim": ssim,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output_root / "restored_vs_gt_metrics.csv", index=False)
    summary = {metric: float(df[metric].mean()) for metric in ["mae", "mse", "psnr", "ssim"]}
    summary["n"] = int(len(df))
    summary["by_class"] = {
        category: {metric: float(value) for metric, value in values.items()}
        for category, values in df.groupby("category")[["psnr", "ssim"]].mean().to_dict("index").items()
    }
    (output_root / "restored_vs_gt_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    if args.val_fraction <= 0 or args.val_fraction >= 1:
        raise ValueError("--val-fraction must be in (0, 1)")
    args.output_root.mkdir(parents=True, exist_ok=True)

    excluded = set(args.exclude_categories)
    benchmark_df, benchmark_ids_all = trainer.load_benchmark_ids(args.benchmark_metrics)
    benchmark_df = filter_categories(benchmark_df, excluded)
    benchmark_ids = set(zip(benchmark_df["category"].astype(str), benchmark_df["image_name"].astype(str)))

    paired_df_all, _ = trainer.collect_paired_samples(args.fog_root, args.gt_root, benchmark_ids_all)
    paired_df = filter_categories(paired_df_all, excluded)
    class_names = sorted(paired_df["category"].unique().tolist())
    if not class_names:
        raise RuntimeError("No classes remain after filtering")

    paired_df.to_csv(args.output_root / "paired_source_manifest_five_class.csv", index=False)

    clear_manifest, split_counts, class_to_idx, idx_to_class = trainer.build_condition_manifest(
        paired_df, class_names, "clear_targets", args.seed, args.val_fraction
    )
    foggy_manifest, foggy_split_counts, _, _ = trainer.build_condition_manifest(
        paired_df, class_names, "foggy_inputs", args.seed, args.val_fraction
    )
    split_description = (
        "same fog-chamber benchmark held-out image IDs after excluding "
        + ", ".join(sorted(excluded))
    )
    split_counts["test_definition"] = split_description
    foggy_split_counts["test_definition"] = split_description
    validate_five_class_split(clear_manifest, len(benchmark_df))
    validate_five_class_split(foggy_manifest, len(benchmark_df))

    clear_manifest.to_csv(args.output_root / "clear_targets_manifest.csv", index=False)
    foggy_manifest.to_csv(args.output_root / "foggy_inputs_manifest.csv", index=False)
    (args.output_root / "split_counts.json").write_text(json.dumps(split_counts, indent=2), encoding="utf-8")

    if args.dry_run:
        print(json.dumps({"dry_run": True, "classes": class_names, "split_counts": split_counts}, indent=2))
        return

    clear_run_dir = trainer.train_condition(
        "clear_targets",
        clear_manifest,
        split_counts,
        class_to_idx,
        idx_to_class,
        args,
    )
    checkpoint_path = clear_run_dir / "best_model.pt"

    eval_root = args.output_root / "frozen_clear_classifier_evaluations"
    clear_metrics = evaluate_checkpoint_on_manifest(
        "clear_targets",
        checkpoint_path,
        clear_manifest[clear_manifest["split"] == "test"].copy(),
        eval_root,
        args.batch_size,
        args.num_workers,
    )
    foggy_metrics = evaluate_checkpoint_on_manifest(
        "foggy_inputs",
        checkpoint_path,
        foggy_manifest[foggy_manifest["split"] == "test"].copy(),
        eval_root,
        args.batch_size,
        args.num_workers,
    )

    restored_root = args.output_root / "restored_images_png"
    restored_manifest = build_restored_manifest(foggy_manifest, restored_root)
    restored_manifest.to_csv(args.output_root / "restored_test_manifest.csv", index=False)
    restore_summary = restore_eval.restore_test_images(
        restored_manifest=restored_manifest,
        checkpoint_path=args.nafnet_checkpoint,
        restored_root=restored_root,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        overwrite=args.overwrite_restored,
    )
    restored_image_metrics = compute_restoration_metrics(restored_manifest, args.output_root)
    restored_metrics = evaluate_checkpoint_on_manifest(
        "nafnet_restored_outputs",
        checkpoint_path,
        restored_manifest,
        eval_root,
        args.batch_size,
        args.num_workers,
    )

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Five-class semantic-preservation check with one clear-trained frozen ResNet-50.",
        "excluded_categories": sorted(excluded),
        "class_names": class_names,
        "benchmark_metrics": str(args.benchmark_metrics),
        "fog_root": str(args.fog_root),
        "gt_root": str(args.gt_root),
        "split_definition": split_description,
        "split_counts": split_counts,
        "foggy_split_counts": foggy_split_counts,
        "clear_classifier_run": str(clear_run_dir),
        "clear_classifier_checkpoint": str(checkpoint_path),
        "restore_summary": restore_summary,
        "restored_image_metrics": restored_image_metrics,
        "classification_metrics": {
            "clear_targets": clear_metrics,
            "foggy_inputs": foggy_metrics,
            "nafnet_restored_outputs": restored_metrics,
        },
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows = [
        {"condition": key, **value}
        for key, value in summary["classification_metrics"].items()
    ]
    pd.DataFrame(rows).to_csv(args.output_root / "classification_summary_table.csv", index=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
