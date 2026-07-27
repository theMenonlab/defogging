#!/usr/bin/env python3
"""Fine-tune one benchmark model on randomized or depth-sensitive synthetic fog."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from skimage.metrics import structural_similarity
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fog_synthesis import FogPreset, jitter_preset, load_preset, synthesize_fog


VALID_SUFFIXES = {".jpg", ".jpeg", ".png"}


def train_one(*_args: object, **_kwargs: object) -> None:
    """Compatibility symbol for trusted legacy checkpoints.

    The old argparse namespace stored its ``func=train_one`` callback inside
    each checkpoint. PyTorch therefore looks up ``__main__.train_one`` while
    unpickling even though this function is never called.
    """
    raise RuntimeError("Legacy checkpoint callback must never be called")


def stable_seed(path: Path, base_seed: int) -> int:
    digest = hashlib.blake2b(str(path).encode("utf-8"), digest_size=8).digest()
    return int((int.from_bytes(digest, "little") + base_seed * 1_000_003) % (2**32 - 1))


def collect_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VALID_SUFFIXES)


def resolve_splits(mapillary_root: Path) -> dict[str, Path]:
    root = mapillary_root
    if not (root / "training" / "images").is_dir() and (root / "Mapillary Vistas").is_dir():
        root = root / "Mapillary Vistas"
    split_dirs = {
        "train": root / "training" / "images",
        "val": root / "validation" / "images",
        "test": root / "testing" / "images",
    }
    missing = [str(path) for path in split_dirs.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing Mapillary image directories: " + ", ".join(missing))
    return split_dirs


def resize_pair(rgb: np.ndarray, depth: np.ndarray | None, patch_size: int) -> tuple[np.ndarray, np.ndarray | None]:
    height, width = rgb.shape[:2]
    if height >= patch_size and width >= patch_size:
        return rgb, depth
    scale = max(patch_size / height, patch_size / width)
    new_size = (int(math.ceil(width * scale)), int(math.ceil(height * scale)))
    rgb = np.asarray(
        Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), mode="RGB").resize(
            new_size, Image.Resampling.BICUBIC
        ),
        dtype=np.float32,
    ) / 255.0
    if depth is not None:
        depth = np.asarray(Image.fromarray(depth.astype(np.float32)).resize(new_size, Image.Resampling.BILINEAR), dtype=np.float32)
    return rgb, depth


class MapillaryFogDataset(Dataset):
    def __init__(
        self,
        image_root: Path,
        depth_root: Path,
        split_name: str,
        mode: str,
        preset: FogPreset,
        patch_size: int,
        seed: int,
        airlight_jitter: float,
        beta_mult_min: float,
        beta_mult_max: float,
        variation_mult_min: float,
        variation_mult_max: float,
        light_fog_prob: float,
        identity_prob: float,
        max_images: int | None = None,
    ) -> None:
        self.image_root = image_root
        self.depth_root = depth_root
        self.split_name = split_name
        self.mode = mode
        self.preset = preset
        self.patch_size = patch_size
        self.seed = seed
        self.airlight_jitter = airlight_jitter
        self.beta_mult_min = beta_mult_min
        self.beta_mult_max = beta_mult_max
        self.variation_mult_min = variation_mult_min
        self.variation_mult_max = variation_mult_max
        self.light_fog_prob = light_fog_prob
        self.identity_prob = identity_prob
        self.paths = collect_images(image_root)
        if max_images is not None:
            rng = random.Random(seed)
            rng.shuffle(self.paths)
            self.paths = sorted(self.paths[:max_images])
        if not self.paths:
            raise ValueError(f"No images found in {image_root}")

    def __len__(self) -> int:
        return len(self.paths)

    def _depth_path(self, image_path: Path) -> Path:
        return self.depth_root / self.image_root.parent.name / image_path.relative_to(self.image_root).with_suffix(".npy")

    def __getitem__(self, index: int) -> dict[str, object]:
        path = self.paths[index]
        extra_seed = int(np.random.randint(0, 1_000_000_000)) if self.split_name == "train" else 0
        rng = np.random.default_rng((stable_seed(path, self.seed) + extra_seed) % (2**32 - 1))
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        depth = None
        if self.mode == "andrew_depth":
            depth_path = self._depth_path(path)
            if not depth_path.is_file():
                raise FileNotFoundError(f"Missing depth map: {depth_path}")
            depth = np.load(depth_path, mmap_mode="r").astype(np.float32)
            if depth.shape != rgb.shape[:2]:
                depth = np.asarray(
                    Image.fromarray(depth).resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.BILINEAR),
                    dtype=np.float32,
                )
        rgb, depth = resize_pair(rgb, depth, self.patch_size)
        height, width = rgb.shape[:2]
        if self.split_name == "train":
            top = int(rng.integers(0, height - self.patch_size + 1))
            left = int(rng.integers(0, width - self.patch_size + 1))
        else:
            top = (height - self.patch_size) // 2
            left = (width - self.patch_size) // 2
        clear = rgb[top : top + self.patch_size, left : left + self.patch_size]
        depth_crop = None if depth is None else depth[top : top + self.patch_size, left : left + self.patch_size]

        if self.split_name == "train":
            sample_preset, identity = jitter_preset(
                self.preset,
                rng,
                beta_mult_min=self.beta_mult_min,
                beta_mult_max=self.beta_mult_max,
                variation_mult_min=self.variation_mult_min,
                variation_mult_max=self.variation_mult_max,
                airlight_jitter=self.airlight_jitter,
                light_fog_prob=self.light_fog_prob,
                identity_prob=self.identity_prob,
            )
        else:
            sample_preset, identity = jitter_preset(
                self.preset,
                rng,
                beta_mult_min=1.0,
                beta_mult_max=1.0,
                variation_mult_min=1.0,
                variation_mult_max=1.0,
                light_fog_prob=0.0,
                identity_prob=0.0,
            )
        if identity:
            foggy = clear.copy()
            fog_stats = {"transmission_mean": 1.0}
        else:
            foggy, fog_stats = synthesize_fog(clear, sample_preset, self.mode, depth_crop)
        if self.split_name == "train" and float(rng.random()) < 0.5:
            clear = np.flip(clear, axis=1).copy()
            foggy = np.flip(foggy, axis=1).copy()
        return {
            "input": torch.from_numpy(np.moveaxis(foggy, -1, 0)).float(),
            "target": torch.from_numpy(np.moveaxis(clear, -1, 0)).float(),
            "path": str(path),
            "transmission_mean": float(fog_stats["transmission_mean"]),
        }


def import_benchmark(path: Path):
    spec = importlib.util.spec_from_file_location("synthetic_model_registry", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_initialized_model(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, dict[str, object]]:
    benchmark = import_benchmark(args.benchmark_script)
    runs = {row.model_key: row for row in benchmark.load_model_runs(args.run6_root)}
    if args.model_key not in runs:
        raise KeyError(f"Unknown model key {args.model_key}")
    model = benchmark.build_model_for_run(args.run6_root, runs[args.model_key])
    checkpoint_path = args.init_checkpoint_root / args.model_key / "final.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Required chamber checkpoint is missing: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)
    return model.to(device), {
        "path": str(checkpoint_path),
        "strict": True,
        "source_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
    }


def color_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred.mean(dim=(2, 3)) - target.mean(dim=(2, 3))))


def residual_tv_loss(pred: torch.Tensor, inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    residual = pred - inp
    luminance = target.mean(dim=1, keepdim=True)
    gx = torch.mean(torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1]), dim=1, keepdim=True)
    gy = torch.mean(torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :]), dim=1, keepdim=True)
    mask_x = (luminance[:, :, :, 1:] > 0.62).float() * torch.exp(-18.0 * gx)
    mask_y = (luminance[:, :, 1:, :] > 0.62).float() * torch.exp(-18.0 * gy)
    tv_x = torch.abs(residual[:, :, :, 1:] - residual[:, :, :, :-1]) * mask_x
    tv_y = torch.abs(residual[:, :, 1:, :] - residual[:, :, :-1, :]) * mask_y
    return 0.5 * (tv_x.mean() + tv_y.mean())


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int) -> dict[str, float | int]:
    rows: list[tuple[float, float, float]] = []
    model.eval()
    with torch.inference_mode():
        for batch_index, batch in enumerate(tqdm(loader, desc="evaluate")):
            if batch_index >= max_batches:
                break
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            pred = model(inputs).clamp(0.0, 1.0)
            l1 = float(torch.mean(torch.abs(pred - targets)).item())
            pred_np = pred[0].detach().cpu().permute(1, 2, 0).numpy()
            target_np = targets[0].detach().cpu().permute(1, 2, 0).numpy()
            mse = float(np.mean((pred_np - target_np) ** 2))
            psnr = 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)
            ssim = float(structural_similarity(target_np, pred_np, channel_axis=2, data_range=1.0))
            rows.append((l1, psnr, ssim))
    return {
        "batches": len(rows),
        "mean_l1": float(np.mean([row[0] for row in rows])),
        "mean_psnr": float(np.mean([row[1] for row in rows])),
        "mean_ssim": float(np.mean([row[2] for row in rows])),
    }


def save_checkpoint(path: Path, model: nn.Module, optimizer: AdamW, config: dict[str, object], step: int) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "optimizer_step": step,
            "config": config,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--fog-mode", choices=["randomized", "andrew_depth"], required=True)
    parser.add_argument("--mapillary-root", required=True, type=Path)
    parser.add_argument("--depth-root", required=True, type=Path)
    parser.add_argument("--preset-json", required=True, type=Path)
    parser.add_argument("--benchmark-script", required=True, type=Path)
    parser.add_argument("--run6-root", required=True, type=Path)
    parser.add_argument("--init-checkpoint-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--crop-size", type=int, required=True)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-optimizer-steps", type=int, default=2600)
    parser.add_argument("--max-val-batches", type=int, default=300)
    parser.add_argument("--max-test-batches", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--color-loss-weight", type=float, default=0.02)
    parser.add_argument("--residual-tv-weight", type=float, default=0.008)
    parser.add_argument("--airlight-jitter", type=float, default=0.03)
    parser.add_argument("--beta-mult-min", type=float, default=0.55)
    parser.add_argument("--beta-mult-max", type=float, default=1.40)
    parser.add_argument("--variation-mult-min", type=float, default=0.60)
    parser.add_argument("--variation-mult-max", type=float, default=1.15)
    parser.add_argument("--light-fog-prob", type=float, default=0.30)
    parser.add_argument("--identity-prob", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=734)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument(
        "--max-images-per-split",
        type=int,
        default=None,
        help="Optional deterministic subset size for validation or development runs.",
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=0.0,
        help="Stop training cleanly after this many seconds, then save and evaluate. Zero disables the limit.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.max_optimizer_steps = min(args.max_optimizer_steps, 1)
        args.max_val_batches = min(args.max_val_batches, 1)
        args.max_test_batches = min(args.max_test_batches, 1)
        args.num_workers = 0
        if args.max_images_per_split is None:
            args.max_images_per_split = 4
    if args.max_wall_seconds < 0:
        raise ValueError("--max-wall-seconds must be nonnegative")
    if args.max_images_per_split is not None and args.max_images_per_split <= 0:
        raise ValueError("--max-images-per-split must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = args.output_root / args.fog_mode / args.model_key
    if args.smoke:
        run_dir = args.output_root / "smoke" / args.fog_mode / args.model_key
    if run_dir.exists() and any(run_dir.iterdir()):
        if not args.force:
            raise FileExistsError(f"Run directory is nonempty: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    preset = load_preset(args.preset_json)
    split_dirs = resolve_splits(args.mapillary_root)
    datasets = {
        split: MapillaryFogDataset(
            image_root=path,
            depth_root=args.depth_root,
            split_name=split,
            mode=args.fog_mode,
            preset=preset,
            patch_size=args.crop_size,
            seed=args.seed,
            airlight_jitter=args.airlight_jitter,
            beta_mult_min=args.beta_mult_min,
            beta_mult_max=args.beta_mult_max,
            variation_mult_min=args.variation_mult_min,
            variation_mult_max=args.variation_mult_max,
            light_fog_prob=args.light_fog_prob,
            identity_prob=args.identity_prob,
            max_images=args.max_images_per_split,
        )
        for split, path in split_dirs.items()
    }
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.micro_batch,
            shuffle=True,
            generator=generator,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(datasets["val"], batch_size=1, shuffle=False, num_workers=min(2, args.num_workers)),
        "test": DataLoader(datasets["test"], batch_size=1, shuffle=False, num_workers=min(2, args.num_workers)),
    }
    maximum_available_steps = args.epochs * (len(loaders["train"]) // args.accumulation_steps)
    if args.max_optimizer_steps > maximum_available_steps:
        raise ValueError(
            f"Requested {args.max_optimizer_steps} optimizer steps, but {args.epochs} epochs provide "
            f"at most {maximum_available_steps}"
        )
    required_micro_batches = args.max_optimizer_steps * args.accumulation_steps

    model, init_info = build_initialized_model(args, device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    config: dict[str, object] = {
        "timestamp": datetime.now().isoformat(),
        "model_key": args.model_key,
        "fog_mode": args.fog_mode,
        "depth_normalization": None,
        "andrew_depth_formula": "beta=beta_mean*(1+paint_weight*raw_zoe_depth), clipped to [0.03,8], then T=exp(-beta*base_depth)",
        "mapillary_root": str(args.mapillary_root),
        "depth_root": str(args.depth_root),
        "split_counts": {key: len(value) for key, value in datasets.items()},
        "max_images_per_split": args.max_images_per_split,
        "preset": preset.__dict__,
        "crop_size": args.crop_size,
        "micro_batch": args.micro_batch,
        "accumulation_steps": args.accumulation_steps,
        "effective_batch_size": args.micro_batch * args.accumulation_steps,
        "epochs": args.epochs,
        "max_optimizer_steps": args.max_optimizer_steps,
        "max_val_batches": args.max_val_batches,
        "max_test_batches": args.max_test_batches,
        "max_wall_seconds": args.max_wall_seconds,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "color_loss_weight": args.color_loss_weight,
        "residual_tv_weight": args.residual_tv_weight,
        "airlight_jitter": args.airlight_jitter,
        "beta_mult_min": args.beta_mult_min,
        "beta_mult_max": args.beta_mult_max,
        "variation_mult_min": args.variation_mult_min,
        "variation_mult_max": args.variation_mult_max,
        "light_fog_prob": args.light_fog_prob,
        "identity_prob": args.identity_prob,
        "scheduler_behavior": "constant learning rate during the requested optimizer-step budget",
        "seed": args.seed,
        "init_checkpoint": init_info,
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": str(device),
        "smoke": args.smoke,
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    start = time.time()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    micro_batch_count = 0
    loss_sum = 0.0
    completed_epochs = 0
    stop_reason: str | None = None
    progress = tqdm(total=required_micro_batches, desc=f"{args.model_key} {args.fog_mode}")
    try:
        for _epoch_index in range(args.epochs):
            epoch_micro_batches = 0
            for batch in loaders["train"]:
                inputs = batch["input"].to(device, non_blocking=True)
                targets = batch["target"].to(device, non_blocking=True)
                with autocast(enabled=use_amp):
                    pred = model(inputs)
                    base = torch.mean(torch.abs(pred - targets))
                    color = color_loss(pred, targets)
                    tv = residual_tv_loss(pred, inputs, targets)
                    loss = base + args.color_loss_weight * color + args.residual_tv_weight * tv
                    scaled_loss = loss / args.accumulation_steps
                scaler.scale(scaled_loss).backward()
                micro_batch_count += 1
                epoch_micro_batches += 1
                progress.update(1)
                loss_sum += float(loss.item())
                if micro_batch_count % args.accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1
                    if optimizer_step % args.checkpoint_every == 0:
                        save_checkpoint(run_dir / "last.pth", model, optimizer, config, optimizer_step)
                    if optimizer_step >= args.max_optimizer_steps:
                        stop_reason = "max_optimizer_steps"
                    elif args.max_wall_seconds > 0 and time.time() - start >= args.max_wall_seconds:
                        stop_reason = "max_wall_seconds"
                if stop_reason is not None:
                    break
            if epoch_micro_batches == len(loaders["train"]):
                completed_epochs += 1
            if stop_reason is not None:
                break
    finally:
        progress.close()
    if stop_reason is None and optimizer_step != args.max_optimizer_steps:
        raise RuntimeError(f"Stopped after {optimizer_step} optimizer steps; expected {args.max_optimizer_steps}")
    save_checkpoint(run_dir / "final.pth", model, optimizer, config, optimizer_step)
    val_metrics = evaluate(model, loaders["val"], device, args.max_val_batches)
    test_metrics = evaluate(model, loaders["test"], device, args.max_test_batches)
    summary = {
        "model_key": args.model_key,
        "fog_mode": args.fog_mode,
        "optimizer_steps": optimizer_step,
        "requested_optimizer_steps": args.max_optimizer_steps,
        "completed_epochs": completed_epochs,
        "requested_epochs": args.epochs,
        "stop_reason": stop_reason,
        "micro_batches": micro_batch_count,
        "mean_train_loss": loss_sum / max(1, micro_batch_count),
        "validation": val_metrics,
        "test": test_metrics,
        "elapsed_seconds": time.time() - start,
        "timestamp": datetime.now().isoformat(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
