#!/usr/bin/env python3
"""Train supervised NAFNet on the NTIRE 2026 nighttime dehazing train GT pairs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_real_haze_nafnet import (
    DEVICE,
    ROOT,
    RUNS_DIR,
    PairRecord,
    PatchPairDataset,
    build_model,
    evaluate_fullres,
    maybe_load_init,
    save_checkpoint,
)

NTIRE_DIR = ROOT / "ntire"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="nafnet_ntire26_supervised_ft_20260601")
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--patches-per-image", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=242)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--middle-blocks", type=int, default=12)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=int, default=96)
    parser.add_argument("--patience", type=int, default=14)
    return parser.parse_args()


def collect_ntire_pairs() -> list[PairRecord]:
    records: list[PairRecord] = []
    for hazy in sorted((NTIRE_DIR / "ntire26_train_inputs").glob("*_NTHazy.png")):
        image_id = hazy.stem.replace("_NTHazy", "")
        gt = NTIRE_DIR / "ntire26_train_gt" / f"{image_id}_GT.png"
        if not gt.exists():
            raise FileNotFoundError(f"Missing NTIRE GT for {hazy}: {gt}")
        number = int(image_id)
        split = "train" if number <= 20 else "val" if number <= 23 else "test"
        records.append(PairRecord("NTIRE26-NH", image_id, hazy, gt, split))
    return records


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = RUNS_DIR / args.run_name
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=False)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    records = collect_ntire_pairs()
    train_records = [r for r in records if r.split == "train"]
    val_records = [r for r in records if r.split == "val"]
    test_records = [r for r in records if r.split == "test"]
    if not train_records or not val_records or not test_records:
        raise RuntimeError(
            f"Bad NTIRE split counts: train={len(train_records)} val={len(val_records)} test={len(test_records)}"
        )

    enc_blocks = [2, 2, 4, 8]
    dec_blocks = [2, 2, 2, 2]
    run_config = {
        "model_name": "nafnet_ntire26_supervised_finetune",
        "model_type": "residual_rgb",
        "dataset": "ntire26_nighttime_train_gt",
        "device": str(DEVICE),
        "split_policy": "01-20 train, 21-23 validation, 24-25 held-out local test",
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
        "init_checkpoint": str(args.init_checkpoint),
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
    val_ds = PatchPairDataset(val_records, args.patch_size, max(8, args.patches_per_image // 2), augment=False, seed=args.seed + 999)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(args.width, args.middle_blocks, enc_blocks, dec_blocks).to(DEVICE)
    run_config["init_loaded"] = maybe_load_init(model, str(args.init_checkpoint))
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    criterion = nn.L1Loss()
    scaler = GradScaler(enabled=torch.cuda.is_available())

    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_epoch = 0
    patience_left = args.patience
    fields = ["epoch", "train_l1", "val_l1", "val_psnr_proxy", "lr", "seconds"]
    with (out_dir / "train_history.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"{args.run_name} epoch {epoch}/{args.epochs}", leave=False)
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
        row = {
            "epoch": float(epoch),
            "train_l1": float(np.mean(train_losses)),
            "val_l1": float(np.mean(val_losses)),
            "val_psnr_proxy": float(np.mean(val_psnrs)),
            "lr": float(scheduler.get_last_lr()[0]),
            "seconds": float(time.time() - start),
        }
        history.append(row)
        with (out_dir / "train_history.csv").open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
        print(
            f"{args.run_name} epoch {epoch:03d}: "
            f"train_l1={row['train_l1']:.5f} "
            f"val_l1={row['val_l1']:.5f} "
            f"val_psnr_proxy={row['val_psnr_proxy']:.2f} "
            f"lr={row['lr']:.2e}"
        )

        save_checkpoint(model, optimizer, epoch, run_config, ckpt_dir / "last_model.pt")
        if row["val_l1"] < best_val:
            best_val = row["val_l1"]
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
