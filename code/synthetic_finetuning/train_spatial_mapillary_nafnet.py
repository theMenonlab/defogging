#!/usr/bin/env python3
"""Train RGB NAFNet on Mapillary Vistas crops with spatial synthetic fog."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
CODE_DIR = HERE.parents[0] / "general_code"
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(HERE))

from nafnet_arch import NAFNet  # noqa: E402
from spatial_fog_model import SpatialFogPreset, preset_from_json, synthesize_spatial_fog  # noqa: E402


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VALID_SUFFIXES = {".jpg", ".jpeg", ".png"}


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


def parse_block_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_seed(path: Path, base_seed: int, extra: int = 0) -> int:
    digest = hashlib.blake2b(str(path).encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return int((value + base_seed * 1_000_003 + extra) % (2**32 - 1))


def collect_images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VALID_SUFFIXES)


def crop_rgb(rgb: np.ndarray, patch_size: int, rng: np.random.Generator, train: bool) -> np.ndarray:
    height, width = rgb.shape[:2]
    if height < patch_size or width < patch_size:
        scale = max(patch_size / height, patch_size / width)
        new_size = (int(math.ceil(width * scale)), int(math.ceil(height * scale)))
        image = Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), mode="RGB")
        image = image.resize(new_size, Image.Resampling.BICUBIC)
        rgb = np.asarray(image, dtype=np.float32) / 255.0
        height, width = rgb.shape[:2]

    if train:
        top = int(rng.integers(0, max(1, height - patch_size + 1)))
        left = int(rng.integers(0, max(1, width - patch_size + 1)))
    else:
        top = max(0, (height - patch_size) // 2)
        left = max(0, (width - patch_size) // 2)
    return rgb[top : top + patch_size, left : left + patch_size, :]


class MapillarySpatialFogDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        split: str,
        preset: SpatialFogPreset,
        patch_size: int,
        base_seed: int,
        airlight_jitter: float,
        beta_mult_min: float,
        beta_mult_max: float,
        variation_mult_min: float,
        variation_mult_max: float,
        light_fog_prob: float,
        identity_prob: float,
        augment: bool,
    ) -> None:
        if not paths:
            raise ValueError(f"No images for split {split}")
        self.paths = [Path(path) for path in paths]
        self.split = split
        self.preset = preset
        self.patch_size = patch_size
        self.base_seed = base_seed
        self.airlight_jitter = airlight_jitter
        self.beta_mult_min = beta_mult_min
        self.beta_mult_max = beta_mult_max
        self.variation_mult_min = variation_mult_min
        self.variation_mult_max = variation_mult_max
        self.light_fog_prob = light_fog_prob
        self.identity_prob = identity_prob
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def _sample_seed(self, index: int) -> int:
        extra = 0
        if self.split == "train":
            extra = int(np.random.randint(0, 1_000_000_000))
        return stable_seed(self.paths[index], self.base_seed, extra=extra)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | float]:
        path = self.paths[index]
        seed = self._sample_seed(index)
        rng = np.random.default_rng(seed)
        clear_rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        clear_crop = crop_rgb(clear_rgb, self.patch_size, rng, train=self.split == "train")

        if float(rng.random()) < self.identity_prob:
            foggy_crop = clear_crop.copy()
            beta_mult = 0.0
            variation_mult = 0.0
            preset = self.preset
        else:
            if self.split == "train":
                beta_mult = float(rng.uniform(self.beta_mult_min, self.beta_mult_max))
                variation_mult = float(rng.uniform(self.variation_mult_min, self.variation_mult_max))
                if float(rng.random()) < self.light_fog_prob:
                    beta_mult *= float(rng.uniform(0.12, 0.45))
                    variation_mult *= float(rng.uniform(0.25, 0.70))
            else:
                beta_mult = 1.0
                variation_mult = 1.0
            jitter = rng.uniform(-self.airlight_jitter, self.airlight_jitter, size=3).astype(np.float32)
            preset = replace(
                self.preset,
                seed=int(seed % 10_000_000),
                beta_mean=float(self.preset.beta_mean * beta_mult),
                beta_variation=float(self.preset.beta_variation * variation_mult),
                airlight_r=float(np.clip(self.preset.airlight_r + jitter[0], 0.0, 1.0)),
                airlight_g=float(np.clip(self.preset.airlight_g + jitter[1], 0.0, 1.0)),
                airlight_b=float(np.clip(self.preset.airlight_b + jitter[2], 0.0, 1.0)),
            )
            foggy_crop, _field, _fog_amount = synthesize_spatial_fog(clear_crop, preset)

        if self.augment:
            if float(rng.random()) < 0.5:
                clear_crop = np.flip(clear_crop, axis=1).copy()
                foggy_crop = np.flip(foggy_crop, axis=1).copy()
            if float(rng.random()) < 0.15:
                clear_crop = np.flip(clear_crop, axis=0).copy()
                foggy_crop = np.flip(foggy_crop, axis=0).copy()

        return {
            "input": torch.from_numpy(np.moveaxis(foggy_crop, -1, 0)).float(),
            "target": torch.from_numpy(np.moveaxis(clear_crop, -1, 0)).float(),
            "clear_path": str(path),
            "split": self.split,
            "fog_seed": str(seed),
            "airlight_r": float(preset.airlight_r),
            "airlight_g": float(preset.airlight_g),
            "airlight_b": float(preset.airlight_b),
            "beta_mult": float(beta_mult),
            "variation_mult": float(variation_mult),
        }


def build_model(args: argparse.Namespace) -> nn.Module:
    return ResidualNAFNetRGB(
        width=args.width,
        middle_blocks=args.middle_blocks,
        enc_blocks=parse_block_list(args.enc_blocks),
        dec_blocks=parse_block_list(args.dec_blocks),
    )


def maybe_load_checkpoint(model: nn.Module, checkpoint_path: str | None, strict: bool) -> dict[str, object] | None:
    if not checkpoint_path:
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    result = model.load_state_dict(state_dict, strict=strict)
    payload: dict[str, object] = {"checkpoint_path": checkpoint_path, "strict": strict}
    if not strict:
        payload["missing_keys"] = list(result.missing_keys)
        payload["unexpected_keys"] = list(result.unexpected_keys)
    return payload


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    return 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps**2))


def crop_mean_color_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred.mean(dim=(2, 3)) - target.mean(dim=(2, 3))))


def residual_tv_loss(pred: torch.Tensor, inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    residual = pred - inp
    luminance = target.mean(dim=1, keepdim=True)
    grad_x = torch.mean(torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1]), dim=1, keepdim=True)
    grad_y = torch.mean(torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :]), dim=1, keepdim=True)
    mask_x = (luminance[:, :, :, 1:] > 0.62).float() * torch.exp(-18.0 * grad_x)
    mask_y = (luminance[:, :, 1:, :] > 0.62).float() * torch.exp(-18.0 * grad_y)
    tv_x = torch.abs(residual[:, :, :, 1:] - residual[:, :, :, :-1]) * mask_x
    tv_y = torch.abs(residual[:, :, 1:, :] - residual[:, :, :-1, :]) * mask_y
    return 0.5 * (tv_x.mean() + tv_y.mean())


def compute_loss(preds: torch.Tensor, inputs: torch.Tensor, targets: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    if args.loss == "charbonnier":
        base = charbonnier_loss(preds, targets)
    else:
        base = torch.mean(torch.abs(preds - targets))
    color = crop_mean_color_loss(preds, targets)
    tv = residual_tv_loss(preds, inputs, targets)
    total = base + args.color_loss_weight * color + args.residual_tv_weight * tv
    parts = {
        "base_loss": float(base.detach().item()),
        "color_loss": float(color.detach().item()),
        "residual_tv_loss": float(tv.detach().item()),
    }
    return total, parts


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    prefix: str,
    visuals_dir: Path,
    max_visuals: int,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    batches = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc=f"{prefix} eval")):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs = batch["input"].to(DEVICE, non_blocking=True)
            targets = batch["target"].to(DEVICE, non_blocking=True)
            preds = model(inputs)
            loss = criterion(preds, targets)
            total_loss += float(loss.item())
            total_psnr += psnr(preds.clamp(0.0, 1.0), targets)
            batches += 1
            if batch_index < max_visuals:
                tiled = torch.cat([inputs.cpu(), preds.cpu().clamp(0.0, 1.0), targets.cpu()], dim=3)
                save_image(tiled, visuals_dir / f"{prefix}_{batch_index:03d}.png")
    return {f"{prefix}_loss": total_loss / max(1, batches), f"{prefix}_psnr": total_psnr / max(1, batches), f"{prefix}_batches": batches}


def write_manifest(path: Path, split_paths: dict[str, list[Path]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "path"])
        writer.writeheader()
        for split, paths in split_paths.items():
            for image_path in paths:
                writer.writerow({"split": split, "path": str(image_path)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapillary-root", required=True, type=Path)
    parser.add_argument("--preset-json", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--run-name", default="mapillary_spatial_fog_nafnet")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=613)
    parser.add_argument("--airlight-jitter", type=float, default=0.03)
    parser.add_argument("--beta-mult-min", type=float, default=1.0)
    parser.add_argument("--beta-mult-max", type=float, default=1.0)
    parser.add_argument("--variation-mult-min", type=float, default=1.0)
    parser.add_argument("--variation-mult-max", type=float, default=1.0)
    parser.add_argument("--light-fog-prob", type=float, default=0.0)
    parser.add_argument("--identity-prob", type=float, default=0.0)
    parser.add_argument("--loss", choices=["l1", "charbonnier"], default="l1")
    parser.add_argument("--color-loss-weight", type=float, default=0.0)
    parser.add_argument("--residual-tv-weight", type=float, default=0.0)
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-val-images", type=int, default=None)
    parser.add_argument("--max-test-images", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--middle-blocks", type=int, default=12)
    parser.add_argument("--enc-blocks", default="2,2,4,8")
    parser.add_argument("--dec-blocks", default="2,2,2,2")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    mapillary_root = args.mapillary_root
    train_dir = mapillary_root / "training" / "images"
    val_dir = mapillary_root / "validation" / "images"
    test_dir = mapillary_root / "testing" / "images"
    if not train_dir.exists():
        nested = mapillary_root / "Mapillary Vistas"
        train_dir = nested / "training" / "images"
        val_dir = nested / "validation" / "images"
        test_dir = nested / "testing" / "images"

    split_paths = {
        "train": collect_images(train_dir),
        "val": collect_images(val_dir),
        "test": collect_images(test_dir),
    }
    rng = random.Random(args.seed)
    for split, limit in [("train", args.max_train_images), ("val", args.max_val_images), ("test", args.max_test_images)]:
        if limit is not None:
            paths = list(split_paths[split])
            rng.shuffle(paths)
            split_paths[split] = sorted(paths[:limit])

    run_dir = args.out_dir / args.run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Run dir exists: {run_dir}")
    checkpoint_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    visuals_dir = run_dir / "visuals"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)

    preset = preset_from_json(args.preset_json)
    datasets = {
        "train": MapillarySpatialFogDataset(
            split_paths["train"],
            "train",
            preset,
            args.patch_size,
            args.seed,
            args.airlight_jitter,
            args.beta_mult_min,
            args.beta_mult_max,
            args.variation_mult_min,
            args.variation_mult_max,
            args.light_fog_prob,
            args.identity_prob,
            True,
        ),
        "val": MapillarySpatialFogDataset(split_paths["val"], "val", preset, args.patch_size, args.seed, args.airlight_jitter, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, False),
        "test": MapillarySpatialFogDataset(split_paths["test"], "test", preset, args.patch_size, args.seed, args.airlight_jitter, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, False),
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available()),
        "val": DataLoader(datasets["val"], batch_size=1, shuffle=False, num_workers=max(1, min(4, args.num_workers)), pin_memory=torch.cuda.is_available()),
        "test": DataLoader(datasets["test"], batch_size=1, shuffle=False, num_workers=max(1, min(4, args.num_workers)), pin_memory=torch.cuda.is_available()),
    }

    write_manifest(run_dir / "split_manifest.csv", split_paths)
    model = build_model(args).to(DEVICE)
    init_info = maybe_load_checkpoint(model, args.init_checkpoint, args.strict_load)
    criterion = nn.L1Loss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    run_config = {
        "model_name": "nafnet_rgb_fog",
        "model_type": "residual_rgb",
        "device": str(DEVICE),
        "mapillary_root": str(mapillary_root),
        "preset_json": str(args.preset_json),
        "spatial_preset": asdict(preset),
        "airlight_jitter": args.airlight_jitter,
        "beta_mult_min": args.beta_mult_min,
        "beta_mult_max": args.beta_mult_max,
        "variation_mult_min": args.variation_mult_min,
        "variation_mult_max": args.variation_mult_max,
        "light_fog_prob": args.light_fog_prob,
        "identity_prob": args.identity_prob,
        "loss": args.loss,
        "color_loss_weight": args.color_loss_weight,
        "residual_tv_weight": args.residual_tv_weight,
        "per_sample_seed_policy": "stable path hash plus base seed; train samples add per-call random offset",
        "init_checkpoint": args.init_checkpoint,
        "patch_size": args.patch_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "split_counts": {name: len(dataset) for name, dataset in datasets.items()},
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
        "max_test_batches": args.max_test_batches,
        "run_dir": str(run_dir),
        "best_checkpoint": str(checkpoint_dir / "best_model.pt"),
        "last_checkpoint": str(checkpoint_dir / "last_model.pt"),
        "width": args.width,
        "middle_blocks": args.middle_blocks,
        "enc_blocks": parse_block_list(args.enc_blocks),
        "dec_blocks": parse_block_list(args.dec_blocks),
    }
    if init_info is not None:
        run_config["init_info"] = init_info
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)
    print(json.dumps(run_config, indent=2))

    history: list[dict[str, float | int]] = []
    best_val_psnr = -1.0
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        batches = 0
        progress = tqdm(loaders["train"], desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch_index, batch in enumerate(progress):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            inputs = batch["input"].to(DEVICE, non_blocking=True)
            targets = batch["target"].to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                preds = model(inputs)
                loss, loss_parts = compute_loss(preds, inputs, targets, args)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item())
            batches += 1
            progress.set_postfix({"loss": f"{loss.item():.5f}", "base": f"{loss_parts['base_loss']:.5f}"})

        train_loss = running / max(1, batches)
        val_metrics = evaluate(model, loaders["val"], criterion, "val", visuals_dir, max_visuals=8, max_batches=args.max_val_batches)
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["val_loss"],
            "val_psnr": val_metrics["val_psnr"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        with (logs_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(history)
        print(json.dumps(row))
        if row["val_psnr"] > best_val_psnr:
            best_val_psnr = float(row["val_psnr"])
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "metrics": row, "config": run_config}, checkpoint_dir / "best_model.pt")
        if (epoch + 1) % args.save_every == 0:
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "metrics": row, "config": run_config}, checkpoint_dir / f"epoch_{epoch + 1:03d}.pt")

    torch.save({"epoch": len(history), "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "metrics": history[-1], "config": run_config}, checkpoint_dir / "last_model.pt")
    best_state = torch.load(checkpoint_dir / "best_model.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(best_state["model_state_dict"])
    test_metrics = evaluate(model, loaders["test"], criterion, "test", visuals_dir, max_visuals=16, max_batches=args.max_test_batches)
    summary = {
        "run_config": run_config,
        "history": history,
        "best_val_psnr": best_val_psnr,
        "test_metrics": test_metrics,
        "elapsed_seconds": time.time() - start,
        "timestamp": datetime.now().isoformat(),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
