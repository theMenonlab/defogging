#!/usr/bin/env python3
"""Train and evaluate a ResNet-family image classifier on fog_imager aligned data."""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    ResNeXt50_32X4D_Weights,
    ResNet101_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    Wide_ResNet50_2_Weights,
    resnet101,
    resnet34,
    resnet50,
    resnext50_32x4d,
    wide_resnet50_2,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

MODEL_REGISTRY = {
    "resnet34": {"builder": resnet34, "weights": ResNet34_Weights.IMAGENET1K_V1},
    "resnet50": {"builder": resnet50, "weights": ResNet50_Weights.IMAGENET1K_V2},
    "resnet101": {"builder": resnet101, "weights": ResNet101_Weights.IMAGENET1K_V2},
    "resnext50_32x4d": {
        "builder": resnext50_32x4d,
        "weights": ResNeXt50_32X4D_Weights.IMAGENET1K_V2,
    },
    "wide_resnet50_2": {
        "builder": wide_resnet50_2,
        "weights": Wide_ResNet50_2_Weights.IMAGENET1K_V2,
    },
}


class PathImageDataset(Dataset):
    def __init__(self, paths: List[str], labels: List[int], transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        label = self.labels[index]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train image classifier")
    parser.add_argument("--data-dir", required=True, type=Path, help="Input dataset root")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output root directory")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--head-epochs", default=5, type=int)
    parser.add_argument("--finetune-epochs", default=20, type=int)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--num-workers", default=8, type=int)
    parser.add_argument("--img-size", default=224, type=int)
    parser.add_argument("--min-delta", default=0.001, type=float)
    parser.add_argument(
        "--model-name",
        default="resnet50",
        choices=sorted(MODEL_REGISTRY.keys()),
        help="Backbone architecture to train.",
    )
    parser.add_argument(
        "--init-weights",
        default="imagenet",
        choices=["imagenet", "scratch"],
        help="Model initialization mode: pretrained ImageNet or random init.",
    )
    parser.add_argument(
        "--augment-preset",
        default="mild",
        choices=["mild", "standard", "strong"],
        help="Training augmentation strength.",
    )
    parser.add_argument(
        "--head-type",
        default="linear",
        choices=["linear", "mlp"],
        help="Classifier head type attached to backbone.",
    )
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout before classifier.")
    parser.add_argument(
        "--mlp-hidden-ratio",
        default=0.5,
        type=float,
        help="Hidden dimension ratio for MLP head.",
    )
    parser.add_argument(
        "--optimizer",
        default="adamw",
        choices=["adamw", "sgd"],
        help="Optimizer choice for both training stages.",
    )
    parser.add_argument("--head-lr", default=1e-3, type=float)
    parser.add_argument("--finetune-lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum for SGD optimizer.")
    parser.add_argument(
        "--scheduler",
        default="cosine",
        choices=["cosine", "none"],
        help="LR scheduler used during finetune stage.",
    )
    parser.add_argument(
        "--label-smoothing",
        default=0.0,
        type=float,
        help="Cross entropy label smoothing factor.",
    )
    parser.add_argument(
        "--class-weighting",
        default="none",
        choices=["none", "balanced"],
        help="Apply class weighting to the loss.",
    )
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="Mixup alpha (0 disables).")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="CutMix alpha (0 disables).")
    parser.add_argument(
        "--mixup-prob",
        default=0.0,
        type=float,
        help="Probability to apply mixup or cutmix per batch.",
    )
    parser.add_argument(
        "--mixup-switch-prob",
        default=0.5,
        type=float,
        help="If both mixup and cutmix enabled, probability to choose cutmix.",
    )
    parser.add_argument(
        "--random-erasing-prob",
        default=0.0,
        type=float,
        help="Random erasing probability during training.",
    )
    parser.add_argument(
        "--run-tag",
        default="",
        type=str,
        help="Optional run suffix for easier identification.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def discover_samples(data_dir: Path) -> Tuple[List[str], Dict[str, int], List[Tuple[str, int, str]]]:
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class folders found in {data_dir}")

    class_names = [d.name for d in class_dirs]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    samples: List[Tuple[str, int, str]] = []
    per_class_count: Dict[str, int] = {}
    for class_dir in class_dirs:
        class_name = class_dir.name
        files = sorted(
            [
                p
                for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
        )
        per_class_count[class_name] = len(files)
        if len(files) == 0:
            raise ValueError(f"Class '{class_name}' has no supported images")
        label_idx = class_to_idx[class_name]
        for file_path in files:
            samples.append((str(file_path.resolve()), label_idx, class_name))

    if len(class_names) != 6:
        raise ValueError(
            f"Expected 6 classes for this project, found {len(class_names)}: {class_names}"
        )

    print("Discovered classes and counts:")
    for class_name in class_names:
        print(f"  - {class_name}: {per_class_count[class_name]}")
    print(f"Total images discovered: {len(samples)}")
    return class_names, class_to_idx, samples


def stratified_split(
    samples: List[Tuple[str, int, str]], seed: int
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    paths = np.array([s[0] for s in samples], dtype=object)
    labels = np.array([s[1] for s in samples], dtype=np.int64)
    label_names = np.array([s[2] for s in samples], dtype=object)

    first_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, temp_idx = next(first_split.split(paths, labels))

    temp_labels = labels[temp_idx]
    second_split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_rel_idx, test_rel_idx = next(second_split.split(temp_idx, temp_labels))

    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]

    split_labels = np.full(len(samples), "train", dtype=object)
    split_labels[val_idx] = "val"
    split_labels[test_idx] = "test"

    manifest = pd.DataFrame(
        {
            "path": paths,
            "label_idx": labels,
            "label_name": label_names,
            "split": split_labels,
        }
    )
    manifest = manifest.sort_values(["split", "label_name", "path"]).reset_index(drop=True)

    by_split_and_class: Dict[str, Dict[str, int]] = {}
    for split_name in ["train", "val", "test"]:
        subset = manifest[manifest["split"] == split_name]
        counts = subset["label_name"].value_counts().to_dict()
        by_split_and_class[split_name] = {k: int(v) for k, v in sorted(counts.items())}

    split_counts = {
        "total": int(len(manifest)),
        "train": int((manifest["split"] == "train").sum()),
        "val": int((manifest["split"] == "val").sum()),
        "test": int((manifest["split"] == "test").sum()),
        "by_split_and_class": by_split_and_class,
    }
    return manifest, split_counts


def build_dataloaders(
    manifest: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    img_size: int,
    seed: int,
    use_cuda: bool,
    augment_preset: str,
    random_erasing_prob: float,
) -> Dict[str, DataLoader]:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    eval_resize = int(round((256 / 224) * img_size))

    train_ops: List[object] = [
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
    ]
    if augment_preset == "mild":
        train_ops.append(
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.02,
            )
        )
    elif augment_preset == "standard":
        train_ops.extend(
            [
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.04,
                ),
                transforms.RandomRotation(degrees=12),
            ]
        )
    else:
        train_ops.extend(
            [
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.25,
                    hue=0.06,
                ),
                transforms.RandomRotation(degrees=20),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
            ]
        )
    train_ops.extend([transforms.ToTensor(), normalize])
    if random_erasing_prob > 0:
        train_ops.append(
            transforms.RandomErasing(
                p=random_erasing_prob, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"
            )
        )
    train_transform = transforms.Compose(train_ops)
    eval_transform = transforms.Compose(
        [
            transforms.Resize(eval_resize),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    g = torch.Generator()
    g.manual_seed(seed)

    dataloaders: Dict[str, DataLoader] = {}
    for split_name, transform in [("train", train_transform), ("val", eval_transform), ("test", eval_transform)]:
        split_df = manifest[manifest["split"] == split_name]
        dataset = PathImageDataset(
            paths=split_df["path"].tolist(),
            labels=split_df["label_idx"].astype(int).tolist(),
            transform=transform,
        )
        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=(num_workers > 0),
            worker_init_fn=seed_worker,
            generator=g,
        )
    return dataloaders


def build_model(
    model_name: str,
    init_weights: str,
    num_classes: int,
    head_type: str,
    dropout: float,
    mlp_hidden_ratio: float,
) -> nn.Module:
    model_spec = MODEL_REGISTRY[model_name]
    builder = model_spec["builder"]
    if init_weights == "imagenet":
        print(f"Loading torchvision {model_name} pretrained weights")
        model = builder(weights=model_spec["weights"])
    else:
        print(f"Initializing torchvision {model_name} from scratch (weights=None)")
        model = builder(weights=None)

    in_features = model.fc.in_features
    if head_type == "linear":
        if dropout > 0:
            model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
        else:
            model.fc = nn.Linear(in_features, num_classes)
    else:
        hidden_dim = max(64, int(in_features * mlp_hidden_ratio))
        layers: List[nn.Module] = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.extend([nn.Linear(in_features, hidden_dim), nn.ReLU(inplace=True)])
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(hidden_dim, num_classes))
        model.fc = nn.Sequential(*layers)
    return model


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray) -> Dict[str, float]:
    accuracy = float((y_true == y_pred).mean())
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )

    if y_probs.shape[1] >= 2:
        top2_indices = np.argpartition(y_probs, -2, axis=1)[:, -2:]
        top2_accuracy = float(np.mean([y_true[i] in top2_indices[i] for i in range(len(y_true))]))
    else:
        top2_accuracy = accuracy

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "top2_accuracy": float(top2_accuracy),
    }


def rand_bbox(size: Tuple[int, int, int, int], lam: float) -> Tuple[int, int, int, int]:
    _, _, h, w = size
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    x1 = int(np.clip(cx - cut_w // 2, 0, w))
    y1 = int(np.clip(cy - cut_h // 2, 0, h))
    x2 = int(np.clip(cx + cut_w // 2, 0, w))
    y2 = int(np.clip(cy + cut_h // 2, 0, h))
    return x1, y1, x2, y2


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    amp_enabled: bool,
    mixup_alpha: float,
    cutmix_alpha: float,
    mixup_prob: float,
    mixup_switch_prob: float,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    total_samples = 0
    all_targets: List[int] = []
    all_preds: List[int] = []

    for images, targets, _ in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        use_mix = mixup_prob > 0 and (np.random.rand() < mixup_prob)
        if use_mix and (mixup_alpha > 0 or cutmix_alpha > 0):
            use_cutmix = False
            if cutmix_alpha > 0 and mixup_alpha > 0:
                use_cutmix = np.random.rand() < mixup_switch_prob
            elif cutmix_alpha > 0:
                use_cutmix = True

            if use_cutmix:
                lam = np.random.beta(cutmix_alpha, cutmix_alpha)
                rand_index = torch.randperm(images.size(0), device=images.device)
                targets_a = targets
                targets_b = targets[rand_index]
                x1, y1, x2, y2 = rand_bbox(images.size(), lam)
                images[:, :, y1:y2, x1:x2] = images[rand_index, :, y1:y2, x1:x2]
                lam = 1.0 - ((x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2)))
            else:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                rand_index = torch.randperm(images.size(0), device=images.device)
                targets_a = targets
                targets_b = targets[rand_index]
                images = images * lam + images[rand_index] * (1.0 - lam)
        else:
            targets_a = targets
            targets_b = targets
            lam = 1.0

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = model(images)
            if lam >= 1.0:
                loss = criterion(outputs, targets)
            else:
                loss = lam * criterion(outputs, targets_a) + (1.0 - lam) * criterion(
                    outputs, targets_b
                )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        probs = torch.softmax(outputs.detach(), dim=1)
        preds = torch.argmax(probs, dim=1)

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        all_targets.extend(targets.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    epoch_loss = running_loss / max(total_samples, 1)
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    return {"loss": float(epoch_loss), "accuracy": accuracy}


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, object]:
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []
    all_paths: List[str] = []

    with torch.no_grad():
        for images, targets, paths in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(images)
                loss = criterion(outputs, targets)

            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            preds = np.argmax(probs, axis=1)

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            all_targets.extend(targets.detach().cpu().tolist())
            all_preds.extend(preds.tolist())
            all_probs.append(probs)
            all_paths.extend(list(paths))

    y_true = np.array(all_targets, dtype=np.int64)
    y_pred = np.array(all_preds, dtype=np.int64)
    y_probs = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, 0))

    metrics = compute_metrics(y_true, y_pred, y_probs)
    metrics["loss"] = float(running_loss / max(total_samples, 1))
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["y_probs"] = y_probs
    metrics["paths"] = all_paths
    return metrics


def freeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def make_optimizer(
    optimizer_name: str,
    params,
    lr: float,
    weight_decay: float,
    momentum: float,
) -> torch.optim.Optimizer:
    if optimizer_name == "adamw":
        return torch.optim.AdamW(params=params, lr=lr, weight_decay=weight_decay)
    return torch.optim.SGD(
        params=params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    epoch: int,
    stage: str,
    best_val_macro_f1: float,
    class_to_idx: Dict[str, int],
    idx_to_class: Dict[str, str],
    args: argparse.Namespace,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),
        "stage": stage,
        "best_val_macro_f1": float(best_val_macro_f1),
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "args": vars(args),
    }
    torch.save(payload, path)


def save_training_curves(history_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss", color="tab:blue")
    ax1.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss", color="tab:orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(
        history_df["epoch"],
        history_df["val_macro_f1"],
        label="Val Macro-F1",
        color="tab:green",
        linestyle="--",
    )
    ax2.set_ylabel("Macro-F1")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_confusion_outputs(cm: np.ndarray, class_names: List[str], csv_path: Path, png_path: Path) -> None:
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(csv_path)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("Confusion Matrix (Raw Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
    )
    axes[1].set_title("Confusion Matrix (Row-Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    fig.tight_layout()
    fig.savefig(png_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.head_epochs < 0 or args.finetune_epochs < 0:
        raise ValueError("Epoch counts must be >= 0")
    if args.head_epochs + args.finetune_epochs <= 0:
        raise ValueError("At least one training epoch is required")
    if args.dropout < 0 or args.dropout >= 1:
        raise ValueError("--dropout must be in [0, 1)")
    if args.label_smoothing < 0 or args.label_smoothing >= 1:
        raise ValueError("--label-smoothing must be in [0, 1)")
    if args.mlp_hidden_ratio <= 0:
        raise ValueError("--mlp-hidden-ratio must be > 0")
    if args.mixup_alpha < 0 or args.cutmix_alpha < 0:
        raise ValueError("--mixup-alpha and --cutmix-alpha must be >= 0")
    if not (0.0 <= args.mixup_prob <= 1.0):
        raise ValueError("--mixup-prob must be in [0, 1]")
    if not (0.0 <= args.mixup_switch_prob <= 1.0):
        raise ValueError("--mixup-switch-prob must be in [0, 1]")
    if not (0.0 <= args.random_erasing_prob <= 1.0):
        raise ValueError("--random-erasing-prob must be in [0, 1]")

    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir does not exist: {data_dir}")

    base_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    init_tag = "pretrained" if args.init_weights == "imagenet" else "scratch"
    run_id = f"{base_run_id}_{args.model_name}_{init_tag}"
    if args.run_tag:
        run_id = f"{run_id}_{args.run_tag}"
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    amp_enabled = use_cuda

    env_summary = {
        "python": sys.version.replace("\n", " "),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    with (run_dir / "env_summary.json").open("w", encoding="utf-8") as f:
        json.dump(env_summary, f, indent=2)

    with (run_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    class_names, class_to_idx, samples = discover_samples(data_dir)
    idx_to_class = {str(v): k for k, v in class_to_idx.items()}

    manifest_df, split_counts = stratified_split(samples, args.seed)
    manifest_path = run_dir / "split_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    if len(manifest_df) != len(samples):
        raise RuntimeError("Split manifest row count does not match discovered sample count")

    with (run_dir / "class_to_idx.json").open("w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)
    with (run_dir / "idx_to_class.json").open("w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, indent=2)

    dataloaders = build_dataloaders(
        manifest=manifest_df,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        seed=args.seed,
        use_cuda=use_cuda,
        augment_preset=args.augment_preset,
        random_erasing_prob=args.random_erasing_prob,
    )

    model = build_model(
        model_name=args.model_name,
        init_weights=args.init_weights,
        num_classes=len(class_names),
        head_type=args.head_type,
        dropout=args.dropout,
        mlp_hidden_ratio=args.mlp_hidden_ratio,
    )
    model = model.to(device)

    if args.class_weighting == "balanced":
        train_counts = (
            manifest_df[manifest_df["split"] == "train"]["label_idx"]
            .value_counts()
            .sort_index()
        )
        counts = train_counts.values.astype(np.float32)
        total = float(counts.sum())
        weights = total / (len(counts) * counts)
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing, weight=class_weights
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    history_rows: List[Dict[str, object]] = []
    history_path = run_dir / "training_history.csv"
    best_model_path = run_dir / "best_model.pt"
    last_model_path = run_dir / "last_model.pt"

    best_val_macro_f1 = -1.0
    best_epoch = 0
    best_stage = ""
    epoch_counter = 0
    epochs_without_improve = 0

    if args.head_epochs > 0:
        freeze_backbone(model)
        optimizer = make_optimizer(
            optimizer_name=args.optimizer,
            params=[p for p in model.parameters() if p.requires_grad],
            lr=args.head_lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
        )
        for stage_epoch in range(1, args.head_epochs + 1):
            epoch_counter += 1
            started_at = time.time()
            train_stats = train_one_epoch(
                model=model,
                dataloader=dataloaders["train"],
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                scaler=scaler,
                amp_enabled=amp_enabled,
                mixup_alpha=args.mixup_alpha,
                cutmix_alpha=args.cutmix_alpha,
                mixup_prob=args.mixup_prob,
                mixup_switch_prob=args.mixup_switch_prob,
            )
            val_stats = evaluate(
                model=model,
                dataloader=dataloaders["val"],
                criterion=criterion,
                device=device,
                amp_enabled=amp_enabled,
            )

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
                f"[Epoch {epoch_counter:02d}][head] "
                f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
                f"val_macro_f1={row['val_macro_f1']:.4f}"
            )

            if val_stats["macro_f1"] > (best_val_macro_f1 + args.min_delta):
                best_val_macro_f1 = float(val_stats["macro_f1"])
                best_epoch = epoch_counter
                best_stage = "head"
                save_checkpoint(
                    path=best_model_path,
                    model=model,
                    epoch=epoch_counter,
                    stage=best_stage,
                    best_val_macro_f1=best_val_macro_f1,
                    class_to_idx=class_to_idx,
                    idx_to_class=idx_to_class,
                    args=args,
                )

    if args.finetune_epochs > 0:
        unfreeze_all(model)
        optimizer = make_optimizer(
            optimizer_name=args.optimizer,
            params=model.parameters(),
            lr=args.finetune_lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
        )
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs)
            if args.scheduler == "cosine"
            else None
        )
        for stage_epoch in range(1, args.finetune_epochs + 1):
            epoch_counter += 1
            started_at = time.time()

            train_stats = train_one_epoch(
                model=model,
                dataloader=dataloaders["train"],
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                scaler=scaler,
                amp_enabled=amp_enabled,
                mixup_alpha=args.mixup_alpha,
                cutmix_alpha=args.cutmix_alpha,
                mixup_prob=args.mixup_prob,
                mixup_switch_prob=args.mixup_switch_prob,
            )
            val_stats = evaluate(
                model=model,
                dataloader=dataloaders["val"],
                criterion=criterion,
                device=device,
                amp_enabled=amp_enabled,
            )
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
                f"[Epoch {epoch_counter:02d}][finetune] "
                f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
                f"val_macro_f1={row['val_macro_f1']:.4f}"
            )

            if val_stats["macro_f1"] > (best_val_macro_f1 + args.min_delta):
                best_val_macro_f1 = float(val_stats["macro_f1"])
                best_epoch = epoch_counter
                best_stage = "finetune"
                epochs_without_improve = 0
                save_checkpoint(
                    path=best_model_path,
                    model=model,
                    epoch=epoch_counter,
                    stage=best_stage,
                    best_val_macro_f1=best_val_macro_f1,
                    class_to_idx=class_to_idx,
                    idx_to_class=idx_to_class,
                    args=args,
                )
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= args.patience:
                    print(
                        f"Early stopping triggered after {epochs_without_improve} "
                        f"epochs without macro-F1 improvement."
                    )
                    break

    save_checkpoint(
        path=last_model_path,
        model=model,
        epoch=epoch_counter,
        stage="last",
        best_val_macro_f1=best_val_macro_f1,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        args=args,
    )

    if not best_model_path.exists():
        save_checkpoint(
            path=best_model_path,
            model=model,
            epoch=epoch_counter,
            stage="last",
            best_val_macro_f1=best_val_macro_f1,
            class_to_idx=class_to_idx,
            idx_to_class=idx_to_class,
            args=args,
        )
        best_epoch = epoch_counter
        best_stage = "last"

    best_checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_stats = evaluate(
        model=model,
        dataloader=dataloaders["test"],
        criterion=criterion,
        device=device,
        amp_enabled=amp_enabled,
    )

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

    misclassified_df = predictions_df[predictions_df["correct"] == 0].copy()
    misclassified_df.to_csv(run_dir / "misclassified_samples.csv", index=False)

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    per_class_rows = []
    for class_name in class_names:
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

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    save_confusion_outputs(
        cm=cm,
        class_names=class_names,
        csv_path=run_dir / "confusion_matrix.csv",
        png_path=run_dir / "confusion_matrix.png",
    )

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(history_path, index=False)
    save_training_curves(history_df, run_dir / "training_curves.png")

    metrics = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": args.model_name,
        "init_weights": args.init_weights,
        "augment_preset": args.augment_preset,
        "head_type": args.head_type,
        "optimizer": args.optimizer,
        "label_smoothing": args.label_smoothing,
        "class_weighting": args.class_weighting,
        "mixup_alpha": args.mixup_alpha,
        "cutmix_alpha": args.cutmix_alpha,
        "mixup_prob": args.mixup_prob,
        "mixup_switch_prob": args.mixup_switch_prob,
        "random_erasing_prob": args.random_erasing_prob,
        "weight_decay": args.weight_decay,
        "head_lr": args.head_lr,
        "finetune_lr": args.finetune_lr,
        "class_names": class_names,
        "split_counts": split_counts,
        "best_epoch": int(best_epoch),
        "best_stage": best_stage,
        "val_best_macro_f1": float(best_val_macro_f1),
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
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nRun complete.")
    print(f"Run directory: {run_dir}")
    print(
        "Test metrics: "
        f"acc={metrics['test_accuracy']:.4f}, "
        f"macro_f1={metrics['test_macro_f1']:.4f}, "
        f"top2_acc={metrics['test_top2_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
