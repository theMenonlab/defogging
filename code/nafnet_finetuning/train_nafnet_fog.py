#!/usr/bin/env python3
"""Train an RGB NAFNet model for synthetic fog removal."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from nafnet_arch import NAFNet

from fog_dataset import (
    SyntheticFogDataset,
    collect_images,
    load_preset_bank_json,
    load_preset_json,
    split_paths,
    write_split_manifest,
)
from synth_fog_tools import precompute_depth_cache

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NAFNet on synthetic fog pairs.")
    parser.add_argument("--clear-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--fog-level", choices=["light", "medium", "heavy"], default="medium")
    parser.add_argument("--preset-json", default=None, help="Optional fitted fog preset JSON to override fog-level defaults.")
    parser.add_argument("--preset-bank-json", default=None, help="Optional JSON containing multiple fog presets sampled during training/evaluation.")
    parser.add_argument("--model-type", choices=["plain", "residual_rgb"], default="plain")
    parser.add_argument("--init-checkpoint", default=None, help="Optional checkpoint to initialize weights from.")
    parser.add_argument("--strict-load", action="store_true", help="Require an exact checkpoint match when loading init-checkpoint.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--depth-backend", choices=["auto", "midas", "heuristic"], default="auto")
    parser.add_argument("--depth-cache-dir", default=None, help="Optional directory for cached depth maps used by synthetic fog generation.")
    parser.add_argument("--no-depth-precompute", action="store_true", help="Skip up-front depth cache generation and rely on existing cache files or heuristic fallback inside dataset workers.")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--middle-blocks", type=int, default=12)
    parser.add_argument("--enc-blocks", default="2,2,4,8")
    parser.add_argument("--dec-blocks", default="2,2,2,2")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_block_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    base_seed = torch.initial_seed() % (2**32)
    random.seed(base_seed + worker_id)


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


def build_model(model_type: str, width: int, middle_blocks: int, enc_blocks: list[int], dec_blocks: list[int]) -> nn.Module:
    if model_type == "residual_rgb":
        return ResidualNAFNetRGB(width, middle_blocks, enc_blocks, dec_blocks)
    return NAFNet(
        in_channels=3,
        out_channels=3,
        width=width,
        middle_blk_num=middle_blocks,
        enc_blk_nums=enc_blocks,
        dec_blk_nums=dec_blocks,
    )


def maybe_load_checkpoint(model: nn.Module, checkpoint_path: str | None, strict: bool) -> dict[str, object] | None:
    if not checkpoint_path:
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    load_result = model.load_state_dict(state_dict, strict=strict)
    payload = {
        "checkpoint_path": checkpoint_path,
        "strict": strict,
    }
    if not strict:
        payload["missing_keys"] = list(load_result.missing_keys)
        payload["unexpected_keys"] = list(load_result.unexpected_keys)
    return payload


def psnr_from_l1(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def prepare_run_dir(args: argparse.Namespace) -> dict[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"nafnet_{args.fog_level}_{timestamp}"
    run_dir = Path(args.out_dir) / run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Run directory already exists and is non-empty: {run_dir}")
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "visuals").mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "checkpoint_dir": run_dir / "checkpoints",
        "logs_dir": run_dir / "logs",
        "visuals_dir": run_dir / "visuals",
        "run_config": run_dir / "run_config.json",
        "history_csv": run_dir / "logs" / "history.csv",
        "split_manifest": run_dir / "split_manifest.csv",
        "summary_json": run_dir / "summary.json",
        "best_checkpoint": run_dir / "checkpoints" / "best_model.pt",
        "last_checkpoint": run_dir / "checkpoints" / "last_model.pt",
    }


def make_loaders(args: argparse.Namespace, paths: dict[str, Path]) -> tuple[dict[str, SyntheticFogDataset], dict[str, DataLoader]]:
    clear_paths = collect_images(Path(args.clear_dir))
    if args.depth_backend != "heuristic" and not args.no_depth_precompute:
        precompute_depth_cache(
            clear_paths,
            backend=args.depth_backend,
            cache_dir=args.depth_cache_dir,
            description="precompute depth",
        )
    split_map = split_paths(clear_paths, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    write_split_manifest(split_map, paths["split_manifest"])
    if args.preset_json and args.preset_bank_json:
        raise ValueError("Use either --preset-json or --preset-bank-json, not both.")
    fog_preset = load_preset_json(args.preset_json) if args.preset_json else None
    fog_presets = load_preset_bank_json(args.preset_bank_json) if args.preset_bank_json else None

    datasets = {
        "train": SyntheticFogDataset(
            split_map["train"],
            split="train",
            fog_level=args.fog_level,
            fog_preset=fog_preset,
            fog_presets=fog_presets,
            patch_size=args.patch_size,
            augment=True,
            base_seed=args.seed,
            depth_backend=args.depth_backend,
            depth_cache_dir=args.depth_cache_dir,
            allow_model_depth=False,
        ),
        "val": SyntheticFogDataset(
            split_map["val"],
            split="val",
            fog_level=args.fog_level,
            fog_preset=fog_preset,
            fog_presets=fog_presets,
            patch_size=args.patch_size,
            augment=False,
            base_seed=args.seed,
            depth_backend=args.depth_backend,
            depth_cache_dir=args.depth_cache_dir,
            allow_model_depth=False,
        ),
        "test": SyntheticFogDataset(
            split_map["test"],
            split="test",
            fog_level=args.fog_level,
            fog_preset=fog_preset,
            fog_presets=fog_presets,
            patch_size=args.patch_size,
            augment=False,
            base_seed=args.seed,
            depth_backend=args.depth_backend,
            depth_cache_dir=args.depth_cache_dir,
            allow_model_depth=False,
        ),
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=1,
            shuffle=False,
            num_workers=max(1, min(2, args.num_workers)),
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=1,
            shuffle=False,
            num_workers=max(1, min(2, args.num_workers)),
            pin_memory=torch.cuda.is_available(),
        ),
    }
    return datasets, loaders


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, max_visuals: int = 0, visuals_dir: Path | None = None, prefix: str = "val") -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    batches = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc=f"{prefix} eval", leave=False)):
            inputs = batch["input"].to(DEVICE, non_blocking=True)
            targets = batch["target"].to(DEVICE, non_blocking=True)
            preds = model(inputs)
            loss = criterion(preds, targets)
            total_loss += float(loss.item())
            total_psnr += psnr_from_l1(preds, targets)
            batches += 1

            if visuals_dir is not None and batch_index < max_visuals:
                tiled = torch.cat([inputs.cpu(), preds.cpu().clamp(0.0, 1.0), targets.cpu()], dim=3)
                save_image(tiled, visuals_dir / f"{prefix}_{batch_index:03d}.png")

    return {
        f"{prefix}_loss": total_loss / max(1, batches),
        f"{prefix}_psnr": total_psnr / max(1, batches),
    }


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: CosineAnnealingLR, epoch: int, metrics: dict[str, float], run_config: dict[str, object]) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
        "config": run_config,
    }
    torch.save(payload, path)


def write_history(history_path: Path, rows: list[dict[str, float | int]]) -> None:
    fieldnames = ["epoch", "train_loss", "val_loss", "val_psnr", "lr"]
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    paths = prepare_run_dir(args)
    enc_blocks = parse_block_list(args.enc_blocks)
    dec_blocks = parse_block_list(args.dec_blocks)
    datasets, loaders = make_loaders(args, paths)

    model = build_model(args.model_type, args.width, args.middle_blocks, enc_blocks, dec_blocks).to(DEVICE)
    init_info = maybe_load_checkpoint(model, args.init_checkpoint, strict=args.strict_load)
    criterion = nn.L1Loss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    run_config = {
        "model_name": "nafnet_rgb_fog",
        "model_type": args.model_type,
        "device": str(DEVICE),
        "clear_dir": args.clear_dir,
        "fog_level": args.fog_level,
        "preset_json": args.preset_json,
        "preset_bank_json": args.preset_bank_json,
        "init_checkpoint": args.init_checkpoint,
        "patch_size": args.patch_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "depth_backend": args.depth_backend,
        "depth_cache_dir": args.depth_cache_dir,
        "depth_precompute": not args.no_depth_precompute,
        "width": args.width,
        "middle_blocks": args.middle_blocks,
        "enc_blocks": enc_blocks,
        "dec_blocks": dec_blocks,
        "seed": args.seed,
        "split_counts": {name: len(dataset) for name, dataset in datasets.items()},
        "run_dir": str(paths["run_dir"]),
        "best_checkpoint": str(paths["best_checkpoint"]),
        "last_checkpoint": str(paths["last_checkpoint"]),
    }
    if args.preset_json:
        run_config["fitted_preset"] = json.loads(Path(args.preset_json).read_text(encoding="utf-8"))
    if args.preset_bank_json:
        run_config["preset_bank"] = json.loads(Path(args.preset_bank_json).read_text(encoding="utf-8"))
    if init_info is not None:
        run_config["init_info"] = init_info
    with paths["run_config"].open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    print("=" * 80)
    print(f"Training RGB NAFNet on {DEVICE}")
    print(json.dumps(run_config, indent=2))
    print("=" * 80)

    history: list[dict[str, float | int]] = []
    best_val_psnr = -1.0
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        batches = 0
        progress = tqdm(loaders["train"], desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            inputs = batch["input"].to(DEVICE, non_blocking=True)
            targets = batch["target"].to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                preds = model(inputs)
                loss = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            batches += 1
            progress.set_postfix({"loss": f"{loss.item():.5f}"})
            if args.smoke_only and batches >= 2:
                break

        train_loss = running_loss / max(1, batches)
        val_metrics = evaluate(
            model,
            loaders["val"],
            criterion,
            max_visuals=4 if epoch == 0 or epoch + 1 == args.epochs else 0,
            visuals_dir=paths["visuals_dir"],
            prefix="val",
        )
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["val_loss"],
            "val_psnr": val_metrics["val_psnr"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        write_history(paths["history_csv"], history)
        print(json.dumps(row))

        if val_metrics["val_psnr"] > best_val_psnr:
            best_val_psnr = val_metrics["val_psnr"]
            save_checkpoint(paths["best_checkpoint"], model, optimizer, scheduler, epoch + 1, row, run_config)

        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs or args.smoke_only:
            save_checkpoint(
                paths["checkpoint_dir"] / f"epoch_{epoch + 1:03d}.pt",
                model,
                optimizer,
                scheduler,
                epoch + 1,
                row,
                run_config,
            )

        if args.smoke_only:
            break

    save_checkpoint(paths["last_checkpoint"], model, optimizer, scheduler, len(history), history[-1], run_config)

    best_state = torch.load(paths["best_checkpoint"], map_location=DEVICE, weights_only=False)
    model.load_state_dict(best_state["model_state_dict"])
    test_metrics = evaluate(model, loaders["test"], criterion, max_visuals=8, visuals_dir=paths["visuals_dir"], prefix="test")

    summary = {
        "run_config": run_config,
        "history": history,
        "best_val_psnr": best_val_psnr,
        "test_metrics": test_metrics,
        "elapsed_seconds": time.time() - start,
        "timestamp": datetime.now().isoformat(),
    }
    with paths["summary_json"].open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("=" * 80)
    print(json.dumps(summary, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    main()
