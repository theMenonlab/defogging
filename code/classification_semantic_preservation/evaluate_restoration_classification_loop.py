#!/usr/bin/env python3
"""Evaluate ResNet-50 classifiers on fog-chamber NAFNet restored outputs."""

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
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import classification_train_resnet50_base as base  # noqa: E402
from nafnet_arch import NAFNet  # noqa: E402
from infer_nafnet_fog import load_rgb, save_rgb, tiled_inference  # noqa: E402


ROOT = Path(os.environ.get("DEFOG_WORK_ROOT", "."))
CLASS_ROOT = ROOT / "outputs/fog_chamber_classification_benchmark_20260604"
DEFAULT_FOGGY_RUN = CLASS_ROOT / "foggy_inputs/20260604_103224_resnet50_pretrained_benchmark_split"
DEFAULT_CLEAR_RUN = CLASS_ROOT / "clear_targets/20260604_102651_resnet50_pretrained_benchmark_split"
DEFAULT_OUTPUT = ROOT / "outputs/restoration_classification_loop_20260605"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foggy-run", type=Path, default=DEFAULT_FOGGY_RUN)
    parser.add_argument("--clear-run", type=Path, default=DEFAULT_CLEAR_RUN)
    parser.add_argument(
        "--nafnet-checkpoint",
        type=Path,
        default=ROOT / "fog_chamber_nafnet/nafnet_fc_clean_model_state_20260603.pth",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--overwrite-restored", action="store_true")
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict at {path}")
    return checkpoint


def nafnet_config_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    provenance = checkpoint.get("provenance", {})
    return {
        "model_type": "benchmark_sigmoid_rgb",
        "width": int(provenance.get("width", 32)),
        "middle_blocks": int(provenance.get("middle_blocks", 12)),
        "enc_blocks": list(provenance.get("enc_blocks", [2, 2, 4, 8])),
        "dec_blocks": list(provenance.get("dec_blocks", [2, 2, 2, 2])),
    }


class BenchmarkSigmoidNAFNetRGB(nn.Module):
    """Native RGB NAFNet convention used by the 20260603 fog benchmark."""

    def __init__(self, width: int, middle_blocks: int, enc_blocks: list[int], dec_blocks: list[int]) -> None:
        super().__init__()
        self.core = NAFNet(
            in_channels=3,
            out_channels=3,
            width=width,
            middle_blk_num=middle_blocks,
            enc_blk_nums=enc_blocks,
            dec_blk_nums=dec_blocks,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.core(x))


def build_benchmark_nafnet(config: dict[str, Any]) -> nn.Module:
    return BenchmarkSigmoidNAFNetRGB(
        width=int(config["width"]),
        middle_blocks=int(config["middle_blocks"]),
        enc_blocks=[int(x) for x in config["enc_blocks"]],
        dec_blocks=[int(x) for x in config["dec_blocks"]],
    )


def build_restored_manifest(foggy_manifest_path: Path, restored_root: Path) -> pd.DataFrame:
    manifest = pd.read_csv(foggy_manifest_path)
    test_manifest = manifest[manifest["split"] == "test"].copy()
    if len(test_manifest) != 552:
        raise RuntimeError(f"Expected 552 benchmark test rows, found {len(test_manifest)}")

    output_paths: list[str] = []
    for row in test_manifest.itertuples(index=False):
        category_dir = restored_root / str(row.category)
        output_paths.append(str((category_dir / f"{Path(str(row.image_name)).stem}.png").resolve()))
    test_manifest["foggy_path"] = test_manifest["path"]
    test_manifest["path"] = output_paths
    test_manifest["condition"] = "nafnet_restored_outputs"
    return test_manifest.reset_index(drop=True)


def restore_test_images(
    restored_manifest: pd.DataFrame,
    checkpoint_path: Path,
    restored_root: Path,
    tile_size: int,
    tile_overlap: int,
    overwrite: bool,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict at {checkpoint_path}")
    state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    config = nafnet_config_from_checkpoint(checkpoint)
    model = build_benchmark_nafnet(config).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    written = 0
    skipped = 0
    started = time.time()
    for row in tqdm(restored_manifest.itertuples(index=False), total=len(restored_manifest), desc="NAFNet restore"):
        input_path = Path(row.foggy_path)
        output_path = Path(row.path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not overwrite:
            skipped += 1
            continue
        rgb = load_rgb(input_path)
        pred = tiled_inference(model, rgb, tile_size=tile_size, tile_overlap=tile_overlap)
        save_rgb(output_path, pred)
        written += 1

    return {
        "checkpoint": str(checkpoint_path),
        "model_config": config,
        "restored_root": str(restored_root),
        "image_count": int(len(restored_manifest)),
        "written": int(written),
        "skipped_existing": int(skipped),
        "device": str(device),
        "tile_size": int(tile_size),
        "tile_overlap": int(tile_overlap),
        "elapsed_seconds": time.time() - started,
    }


def eval_transform(img_size: int) -> transforms.Compose:
    eval_resize = int(round((256 / 224) * img_size))
    return transforms.Compose(
        [
            transforms.Resize(eval_resize),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def evaluate_classifier(
    classifier_name: str,
    classifier_run: Path,
    restored_manifest: pd.DataFrame,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    checkpoint_path = classifier_run / "best_model.pt"
    checkpoint = load_checkpoint(checkpoint_path)
    args = checkpoint["args"]
    class_to_idx = {str(k): int(v) for k, v in checkpoint["class_to_idx"].items()}
    idx_to_class = {int(v): str(k) for k, v in class_to_idx.items()}

    eval_manifest = restored_manifest.copy()
    eval_manifest["label_idx"] = eval_manifest["label_name"].map(class_to_idx).astype(int)
    eval_manifest["split"] = "test"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    dataset = base.PathImageDataset(
        paths=eval_manifest["path"].tolist(),
        labels=eval_manifest["label_idx"].astype(int).tolist(),
        transform=eval_transform(int(args.get("img_size", 224))),
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
    pred_labels = [idx_to_class[int(i)] for i in y_pred]
    true_labels = [idx_to_class[int(i)] for i in y_true]
    correct = (y_true == y_pred).astype(int)

    condition_dir = output_dir / classifier_name
    condition_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(
        {
            "pair_id": eval_manifest["pair_id"].tolist(),
            "path": stats["paths"],
            "foggy_path": eval_manifest["foggy_path"].tolist(),
            "true_label": true_labels,
            "pred_label": pred_labels,
            "confidence": confidences,
            "correct": correct,
        }
    )
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
        "classifier_name": classifier_name,
        "classifier_run": str(classifier_run),
        "classifier_checkpoint": str(checkpoint_path),
        "restored_condition": "nafnet_restored_outputs",
        "test_count": int(len(eval_manifest)),
        "test_accuracy": float(stats["accuracy"]),
        "test_top2_accuracy": float(stats["top2_accuracy"]),
        "test_macro_precision": float(stats["macro_precision"]),
        "test_macro_recall": float(stats["macro_recall"]),
        "test_macro_f1": float(stats["macro_f1"]),
        "test_loss": float(stats["loss"]),
    }
    (condition_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    restored_root = args.output_root / "restored_images_png"
    restored_manifest = build_restored_manifest(args.foggy_run / "split_manifest.csv", restored_root)
    restored_manifest.to_csv(args.output_root / "restored_test_manifest.csv", index=False)

    restore_summary = restore_test_images(
        restored_manifest=restored_manifest,
        checkpoint_path=args.nafnet_checkpoint,
        restored_root=restored_root,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        overwrite=args.overwrite_restored,
    )

    classifier_metrics = [
        evaluate_classifier(
            "clear_trained_classifier_on_restored",
            args.clear_run,
            restored_manifest,
            args.output_root,
            args.batch_size,
            args.num_workers,
        ),
        evaluate_classifier(
            "foggy_trained_classifier_on_restored",
            args.foggy_run,
            restored_manifest,
            args.output_root,
            args.batch_size,
            args.num_workers,
        ),
    ]

    baseline_metrics = {}
    for name, run_dir in [("clear_targets", args.clear_run), ("foggy_inputs", args.foggy_run)]:
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            baseline_metrics[name] = {
                "run": str(run_dir),
                "test_accuracy": payload.get("test_accuracy"),
                "test_top2_accuracy": payload.get("test_top2_accuracy"),
                "test_macro_f1": payload.get("test_macro_f1"),
            }

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "purpose": "ResNet-50 recognition after NAFNet restoration on the 552-image fog-chamber benchmark held-out split.",
        "split_definition": "same 552 image IDs as fog_chamber_nafnet/metrics.csv",
        "restore_summary": restore_summary,
        "baseline_metrics": baseline_metrics,
        "restored_classifier_metrics": classifier_metrics,
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
