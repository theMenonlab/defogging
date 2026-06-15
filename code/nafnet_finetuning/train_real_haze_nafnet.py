#!/usr/bin/env python3
"""Train supervised NAFNet models on paired real haze datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
OHAZE_DIR = ROOT / "o-haze" / "O-HAZY"
NHHAZE_DIR = ROOT / "nh-haze" / "NH-HAZE"
RUNS_DIR = ROOT / "outputs"

sys.path.insert(0, str(CODE_DIR))
from nafnet_arch import NAFNet  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class PairRecord:
    dataset: str
    image_id: str
    hazy_path: Path
    gt_path: Path
    split: str


class ResidualNAFNetRGB(nn.Module):
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
        return x + self.core(x)


def build_model(width: int, middle_blocks: int, enc_blocks: list[int], dec_blocks: list[int]) -> nn.Module:
    return ResidualNAFNetRGB(width, middle_blocks, enc_blocks, dec_blocks)


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def array_to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.moveaxis(arr, -1, 0)).float()


def tensor_to_array(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def collect_ohaze() -> list[PairRecord]:
    records = []
    for hazy in sorted((OHAZE_DIR / "hazy").iterdir()):
        if hazy.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        image_id = hazy.stem
        gt = OHAZE_DIR / "GT" / hazy.name
        number = int(image_id.split("_")[0])
        split = "train" if number <= 35 else "val" if number <= 40 else "test"
        records.append(PairRecord("O-HAZE", image_id, hazy, gt, split))
    return records


def collect_nhhaze() -> list[PairRecord]:
    records = []
    for hazy in sorted(NHHAZE_DIR.glob("*_hazy.png")):
        image_id = hazy.stem.replace("_hazy", "")
        gt = hazy.with_name(hazy.name.replace("_hazy", "_GT"))
        number = int(image_id)
        split = "train" if number <= 45 else "val" if number <= 50 else "test"
        records.append(PairRecord("NH-HAZE", image_id, hazy, gt, split))
    return records


def collect_records(dataset: str) -> list[PairRecord]:
    if dataset == "ohaze":
        return collect_ohaze()
    if dataset == "nhhaze":
        return collect_nhhaze()
    if dataset == "mixed":
        return collect_ohaze() + collect_nhhaze()
    raise ValueError(f"Unsupported dataset: {dataset}")


class PatchPairDataset(Dataset):
    def __init__(self, records: list[PairRecord], patch_size: int, patches_per_image: int, augment: bool, seed: int) -> None:
        self.records = records
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records) * self.patches_per_image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index // self.patches_per_image]
        hazy = read_rgb(record.hazy_path)
        gt = read_rgb(record.gt_path)
        if hazy.shape != gt.shape:
            gt = np.asarray(
                Image.fromarray((gt * 255).round().astype(np.uint8)).resize(hazy.shape[1::-1], Image.Resampling.LANCZOS),
                dtype=np.float32,
            ) / 255.0

        rng = random.Random(self.seed + index + int(time.time() // 3600) * 1000003)
        h, w = hazy.shape[:2]
        ps = min(self.patch_size, h, w)
        top = rng.randint(0, h - ps) if h > ps else 0
        left = rng.randint(0, w - ps) if w > ps else 0
        hazy_patch = hazy[top : top + ps, left : left + ps].copy()
        gt_patch = gt[top : top + ps, left : left + ps].copy()

        if self.augment:
            if rng.random() < 0.5:
                hazy_patch = hazy_patch[:, ::-1].copy()
                gt_patch = gt_patch[:, ::-1].copy()
            if rng.random() < 0.5:
                hazy_patch = hazy_patch[::-1, :].copy()
                gt_patch = gt_patch[::-1, :].copy()
            if rng.random() < 0.5:
                hazy_patch = np.transpose(hazy_patch, (1, 0, 2)).copy()
                gt_patch = np.transpose(gt_patch, (1, 0, 2)).copy()
        return array_to_tensor(hazy_patch), array_to_tensor(gt_patch)


def iter_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, tile_size - overlap)
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def tiled_inference(model: nn.Module, rgb: np.ndarray, tile_size: int, overlap: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    output = np.zeros((height, width, 3), dtype=np.float32)
    weight = np.zeros((height, width, 1), dtype=np.float32)
    with torch.no_grad():
        for top in iter_starts(height, tile_size, overlap):
            for left in iter_starts(width, tile_size, overlap):
                patch = rgb[top : top + tile_size, left : left + tile_size]
                tensor = array_to_tensor(patch).unsqueeze(0).to(DEVICE)
                pred = tensor_to_array(model(tensor).squeeze(0))
                ph, pw = pred.shape[:2]
                blend = np.ones((ph, pw, 1), dtype=np.float32)
                ramp = min(overlap, ph // 2, pw // 2)
                if ramp > 0:
                    y = np.minimum(np.arange(ph), np.arange(ph)[::-1]).astype(np.float32)
                    x = np.minimum(np.arange(pw), np.arange(pw)[::-1]).astype(np.float32)
                    blend = np.minimum.outer(
                        np.clip(y / max(1, ramp), 0, 1),
                        np.clip(x / max(1, ramp), 0, 1),
                    )[:, :, None]
                    blend = np.clip(blend, 1e-3, 1)
                output[top : top + ph, left : left + pw] += pred * blend
                weight[top : top + ph, left : left + pw] += blend
    return output / np.clip(weight, 1e-6, None)


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.clip(pred, 0, 1)
    target = np.clip(target, 0, 1)
    return {
        "mae": float(np.mean(np.abs(pred - target))),
        "mse": float(np.mean((pred - target) ** 2)),
        "psnr": float(peak_signal_noise_ratio(target, pred, data_range=1.0)),
        "ssim": float(structural_similarity(target, pred, channel_axis=2, data_range=1.0)),
    }


def evaluate_fullres(model: nn.Module, records: list[PairRecord], out_dir: Path, tile_size: int, overlap: int) -> dict[str, object]:
    pred_dir = out_dir / "test_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in tqdm(records, desc=f"eval {out_dir.name}"):
        hazy = read_rgb(record.hazy_path)
        gt = read_rgb(record.gt_path)
        if hazy.shape != gt.shape:
            gt = np.asarray(
                Image.fromarray((gt * 255).round().astype(np.uint8)).resize(hazy.shape[1::-1], Image.Resampling.LANCZOS),
                dtype=np.float32,
            ) / 255.0
        pred = tiled_inference(model, hazy, tile_size, overlap)
        metrics = compute_metrics(pred, gt)
        pred_name = f"{record.dataset}_{record.image_id}.png".replace("/", "_")
        Image.fromarray((np.clip(pred, 0, 1) * 255).round().astype(np.uint8)).save(pred_dir / pred_name)
        rows.append(
            {
                "dataset": record.dataset,
                "image_id": record.image_id,
                "hazy_path": str(record.hazy_path),
                "gt_path": str(record.gt_path),
                "prediction_path": str(pred_dir / pred_name),
                **metrics,
            }
        )

    summary = {
        "n": len(rows),
        "mean_mae": float(np.mean([r["mae"] for r in rows])) if rows else math.nan,
        "mean_mse": float(np.mean([r["mse"] for r in rows])) if rows else math.nan,
        "mean_psnr": float(np.mean([r["psnr"] for r in rows])) if rows else math.nan,
        "mean_ssim": float(np.mean([r["ssim"] for r in rows])) if rows else math.nan,
    }
    with (out_dir / "test_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["dataset", "image_id", "hazy_path", "gt_path", "prediction_path", "mae", "mse", "psnr", "ssim"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "test_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, run_config: dict[str, object], path: Path) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "run_config": run_config,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["nhhaze", "ohaze", "mixed"], required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--patches-per-image", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--middle-blocks", type=int, default=12)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=int, default=96)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--patience", type=int, default=18)
    return parser.parse_args()


def maybe_load_init(model: nn.Module, init_checkpoint: str | None) -> str | None:
    if not init_checkpoint:
        return None
    checkpoint_path = Path(init_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    return str(checkpoint_path)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_name = args.run_name or f"nafnet_supervised_{args.dataset}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = RUNS_DIR / run_name
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=False)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    records = collect_records(args.dataset)
    train_records = [r for r in records if r.split == "train"]
    val_records = [r for r in records if r.split == "val"]
    test_records = [r for r in records if r.split == "test"]
    if not train_records or not val_records or not test_records:
        raise RuntimeError(f"Bad split counts for {args.dataset}: train={len(train_records)} val={len(val_records)} test={len(test_records)}")

    enc_blocks = [2, 2, 4, 8]
    dec_blocks = [2, 2, 2, 2]
    run_config = {
        "model_name": "nafnet_supervised_real_haze",
        "model_type": "residual_rgb",
        "dataset": args.dataset,
        "device": str(DEVICE),
        "patch_size": args.patch_size,
        "patches_per_image": args.patches_per_image,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "width": args.width,
        "middle_blocks": args.middle_blocks,
        "enc_blocks": enc_blocks,
        "dec_blocks": dec_blocks,
        "seed": args.seed,
        "split_counts": {"train": len(train_records), "val": len(val_records), "test": len(test_records)},
        "run_dir": str(out_dir),
        "init_checkpoint": args.init_checkpoint,
    }

    with (out_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)
    with (out_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "image_id", "split", "hazy_path", "gt_path"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "dataset": record.dataset,
                    "image_id": record.image_id,
                    "split": record.split,
                    "hazy_path": record.hazy_path,
                    "gt_path": record.gt_path,
                }
            )

    train_ds = PatchPairDataset(train_records, args.patch_size, args.patches_per_image, augment=True, seed=args.seed)
    val_ds = PatchPairDataset(val_records, args.patch_size, max(4, args.patches_per_image // 2), augment=False, seed=args.seed + 999)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    model = build_model(args.width, args.middle_blocks, enc_blocks, dec_blocks).to(DEVICE)
    init_loaded = maybe_load_init(model, args.init_checkpoint)
    run_config["init_loaded"] = init_loaded
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    criterion = nn.L1Loss()
    scaler = GradScaler(enabled=torch.cuda.is_available())

    history = []
    best_val = float("inf")
    best_epoch = 0
    patience_left = args.patience
    csv_path = out_dir / "train_history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_l1", "val_l1", "val_psnr_proxy", "lr", "seconds"])
        writer.writeheader()

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"{run_name} epoch {epoch}/{args.epochs}", leave=False)
        for hazy, gt in pbar:
            hazy = hazy.to(DEVICE, non_blocking=True)
            gt = gt.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=torch.cuda.is_available()):
                pred = model(hazy).clamp(0, 1)
                loss = criterion(pred, gt)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.item()))
            pbar.set_postfix(l1=f"{train_losses[-1]:.4f}")

        model.eval()
        val_losses = []
        val_psnrs = []
        with torch.no_grad():
            for hazy, gt in val_loader:
                hazy = hazy.to(DEVICE, non_blocking=True)
                gt = gt.to(DEVICE, non_blocking=True)
                pred = model(hazy).clamp(0, 1)
                val_losses.append(float(criterion(pred, gt).item()))
                mse = torch.mean((pred - gt) ** 2, dim=(1, 2, 3)).detach().cpu().numpy()
                val_psnrs.extend([float(-10.0 * math.log10(max(float(x), 1e-10))) for x in mse])

        scheduler.step()
        train_l1 = float(np.mean(train_losses))
        val_l1 = float(np.mean(val_losses))
        val_psnr = float(np.mean(val_psnrs))
        row = {
            "epoch": epoch,
            "train_l1": train_l1,
            "val_l1": val_l1,
            "val_psnr_proxy": val_psnr,
            "lr": float(scheduler.get_last_lr()[0]),
            "seconds": time.time() - start,
        }
        history.append(row)
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", "train_l1", "val_l1", "val_psnr_proxy", "lr", "seconds"])
            writer.writerow(row)
        print(
            f"{run_name} epoch {epoch:03d}: train_l1={train_l1:.5f} "
            f"val_l1={val_l1:.5f} val_psnr_proxy={val_psnr:.2f} lr={row['lr']:.2e}"
        )

        save_checkpoint(model, optimizer, epoch, run_config, ckpt_dir / "last_model.pt")
        if val_l1 < best_val:
            best_val = val_l1
            best_epoch = epoch
            patience_left = args.patience
            save_checkpoint(model, optimizer, epoch, run_config, ckpt_dir / "best_model.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}")
                break

    checkpoint = torch.load(ckpt_dir / "best_model.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_summary = evaluate_fullres(model, test_records, out_dir, args.tile_size, args.tile_overlap)

    summary = {
        "run_config": run_config,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_l1": best_val,
        "test_metrics": test_summary,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    ax.plot([r["epoch"] for r in history], [r["train_l1"] for r in history], label="train L1")
    ax.plot([r["epoch"] for r in history], [r["val_l1"] for r in history], label="val L1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L1 loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(out_dir / "loss_curve.png", dpi=180)
    fig.savefig(out_dir / "loss_curve.pdf")
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
