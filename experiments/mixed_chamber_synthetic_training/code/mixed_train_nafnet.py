#!/usr/bin/env python3
"""Train NAFNet from scratch with an interleaved chamber/synthetic update schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from fog_synthesis import load_preset
from synthetic_finetune_all_models import (
    MapillaryFogDataset,
    color_loss,
    import_benchmark,
    residual_tv_loss,
    resolve_splits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-key", default="nafnet_fc")
    parser.add_argument("--real-fog-root", required=True, type=Path)
    parser.add_argument("--real-gt-root", required=True, type=Path)
    parser.add_argument("--holdout-every", type=int, default=10)
    parser.add_argument("--mapillary-root", required=True, type=Path)
    parser.add_argument("--depth-root", required=True, type=Path)
    parser.add_argument("--preset-json", required=True, type=Path)
    parser.add_argument("--benchmark-script", required=True, type=Path)
    parser.add_argument("--run6-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--real-optimizer-steps", required=True, type=int)
    parser.add_argument("--synthetic-optimizer-steps", required=True, type=int)
    parser.add_argument("--synthetic-accumulation-steps", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=512)
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
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--max-real-eval-samples", type=int, default=0)
    parser.add_argument("--max-synthetic-val-batches", type=int, default=300)
    parser.add_argument("--max-synthetic-test-batches", type=int, default=500)
    parser.add_argument("--max-images-per-split", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def infinite_batches(loader: DataLoader[Any]) -> Iterator[Any]:
    while True:
        yield from loader


def is_synthetic_step(step_index: int, total_steps: int, synthetic_steps: int) -> bool:
    """Evenly interleave exactly synthetic_steps among total_steps updates."""
    before = (step_index * synthetic_steps) // total_steps
    after = ((step_index + 1) * synthetic_steps) // total_steps
    return after > before


def state_dict_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def random_crop_pair(inputs: torch.Tensor, targets: torch.Tensor, crop_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = inputs.shape[-2:]
    if crop_size >= height and crop_size >= width:
        return inputs, targets
    if crop_size > height or crop_size > width:
        raise ValueError(f"Requested {crop_size}px crop from a {height}x{width} real-fog pair")
    top = int(torch.randint(0, height - crop_size + 1, (1,)).item())
    left = int(torch.randint(0, width - crop_size + 1, (1,)).item())
    return (
        inputs[:, :, top : top + crop_size, left : left + crop_size],
        targets[:, :, top : top + crop_size, left : left + crop_size],
    )


def image_metrics(target: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    l1 = float(np.mean(np.abs(pred - target)))
    mse = float(np.mean((pred - target) ** 2))
    psnr = 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)
    ssim = float(structural_similarity(target, pred, channel_axis=2, data_range=1.0))
    return l1, psnr, ssim


def evaluate_real(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    max_samples: int,
) -> dict[str, float | int]:
    rows: list[tuple[float, float, float]] = []
    model.eval()
    with torch.inference_mode():
        for index, (inputs, targets, _meta) in enumerate(tqdm(loader, desc="evaluate chamber holdout")):
            if max_samples > 0 and index >= max_samples:
                break
            pred = model(inputs.to(device, non_blocking=True)).clamp(0.0, 1.0)
            pred_np = pred[0].detach().cpu().permute(1, 2, 0).numpy()
            target_np = targets[0].numpy().transpose(1, 2, 0)
            rows.append(image_metrics(target_np, pred_np))
    return {
        "samples": len(rows),
        "mean_l1": float(np.mean([row[0] for row in rows])),
        "mean_psnr": float(np.mean([row[1] for row in rows])),
        "mean_ssim": float(np.mean([row[2] for row in rows])),
    }


def evaluate_synthetic(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    max_batches: int,
    description: str,
) -> dict[str, float | int]:
    rows: list[tuple[float, float, float]] = []
    model.eval()
    with torch.inference_mode():
        for index, batch in enumerate(tqdm(loader, desc=description)):
            if index >= max_batches:
                break
            targets = batch["target"]
            pred = model(batch["input"].to(device, non_blocking=True)).clamp(0.0, 1.0)
            pred_np = pred[0].detach().cpu().permute(1, 2, 0).numpy()
            target_np = targets[0].numpy().transpose(1, 2, 0)
            rows.append(image_metrics(target_np, pred_np))
    return {
        "batches": len(rows),
        "mean_l1": float(np.mean([row[0] for row in rows])),
        "mean_psnr": float(np.mean([row[1] for row in rows])),
        "mean_ssim": float(np.mean([row[2] for row in rows])),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    config: dict[str, object],
    global_step: int,
    real_steps: int,
    synthetic_steps: int,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "optimizer_step": global_step,
            "real_optimizer_steps": real_steps,
            "synthetic_optimizer_steps": synthetic_steps,
            "config": config,
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.model_key != "nafnet_fc":
        raise ValueError("This controlled ratio experiment is intentionally restricted to nafnet_fc")
    if args.real_optimizer_steps <= 0 or args.synthetic_optimizer_steps <= 0:
        raise ValueError("Both real and synthetic optimizer-step counts must be positive")
    if args.synthetic_accumulation_steps <= 0:
        raise ValueError("--synthetic-accumulation-steps must be positive")
    if args.smoke:
        args.real_optimizer_steps = 2
        args.synthetic_optimizer_steps = 1
        args.synthetic_accumulation_steps = 2
        args.crop_size = min(args.crop_size, 64)
        args.num_workers = 0
        args.max_real_eval_samples = 1
        args.max_synthetic_val_batches = 1
        args.max_synthetic_test_batches = 1
        args.max_images_per_split = 4

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = args.output_root / args.run_name
    if args.smoke:
        run_dir = args.output_root / "smoke" / args.run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        if not args.force:
            raise FileExistsError(f"Run directory is nonempty: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    benchmark = import_benchmark(args.benchmark_script)
    model_runs = {row.model_key: row for row in benchmark.load_model_runs(args.run6_root)}
    if args.model_key not in model_runs:
        raise KeyError(f"Unknown model key: {args.model_key}")

    real_train = benchmark.FogRGBPairedDataset(
        args.real_fog_root,
        args.real_gt_root,
        split="train",
        holdout_every=args.holdout_every,
    )
    real_test = benchmark.FogRGBPairedDataset(
        args.real_fog_root,
        args.real_gt_root,
        split="test",
        holdout_every=args.holdout_every,
    )
    split_dirs = resolve_splits(args.mapillary_root)
    preset = load_preset(args.preset_json)
    synthetic_datasets = {
        split: MapillaryFogDataset(
            image_root=path,
            depth_root=args.depth_root,
            split_name=split,
            mode="randomized",
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

    real_generator = torch.Generator().manual_seed(args.seed + 1)
    synthetic_generator = torch.Generator().manual_seed(args.seed + 2)
    pin_memory = device.type == "cuda"
    real_train_loader = DataLoader(
        real_train,
        batch_size=1,
        shuffle=True,
        generator=real_generator,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    synthetic_train_loader = DataLoader(
        synthetic_datasets["train"],
        batch_size=1,
        shuffle=True,
        generator=synthetic_generator,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    real_test_loader = DataLoader(real_test, batch_size=1, shuffle=False, num_workers=min(2, args.num_workers))
    synthetic_val_loader = DataLoader(
        synthetic_datasets["val"], batch_size=1, shuffle=False, num_workers=min(2, args.num_workers)
    )
    synthetic_test_loader = DataLoader(
        synthetic_datasets["test"], batch_size=1, shuffle=False, num_workers=min(2, args.num_workers)
    )

    model = benchmark.build_model_for_run(args.run6_root, model_runs[args.model_key]).to(device)
    initialization_fingerprint = state_dict_sha256(model)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    total_steps = args.real_optimizer_steps + args.synthetic_optimizer_steps
    ratio = args.synthetic_optimizer_steps / args.real_optimizer_steps
    config: dict[str, object] = {
        "timestamp": datetime.now().isoformat(),
        "run_name": args.run_name,
        "model_key": args.model_key,
        "initialization": "fresh model initialization from the benchmark registry; no chamber checkpoint loaded",
        "initialization_fingerprint_sha256": initialization_fingerprint,
        "schedule": "deterministic even interleaving using integer error accumulation",
        "real_optimizer_steps": args.real_optimizer_steps,
        "synthetic_optimizer_steps": args.synthetic_optimizer_steps,
        "total_optimizer_steps": total_steps,
        "synthetic_to_real_update_ratio": ratio,
        "synthetic_fraction_of_all_updates": args.synthetic_optimizer_steps / total_steps,
        "synthetic_accumulation_steps": args.synthetic_accumulation_steps,
        "real_images_per_optimizer_step": 1,
        "synthetic_images_per_optimizer_step": args.synthetic_accumulation_steps,
        "real_training_pairs": len(real_train),
        "real_holdout_pairs": len(real_test),
        "real_fog_root": str(args.real_fog_root),
        "real_gt_root": str(args.real_gt_root),
        "holdout_every": args.holdout_every,
        "mapillary_root": str(args.mapillary_root),
        "mapillary_split_counts": {key: len(value) for key, value in synthetic_datasets.items()},
        "preset": preset.__dict__,
        "fog_mode": "randomized",
        "crop_size": args.crop_size,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "real_loss": "L1",
        "synthetic_loss": "L1 + color_weight*color_loss + residual_tv_weight*residual_tv_loss",
        "color_loss_weight": args.color_loss_weight,
        "residual_tv_weight": args.residual_tv_weight,
        "airlight_jitter": args.airlight_jitter,
        "beta_mult_min": args.beta_mult_min,
        "beta_mult_max": args.beta_mult_max,
        "variation_mult_min": args.variation_mult_min,
        "variation_mult_max": args.variation_mult_max,
        "light_fog_prob": args.light_fog_prob,
        "identity_prob": args.identity_prob,
        "seed": args.seed,
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "smoke": args.smoke,
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    real_iterator = infinite_batches(real_train_loader)
    synthetic_iterator = infinite_batches(synthetic_train_loader)
    real_steps = 0
    synthetic_steps = 0
    real_loss_sum = 0.0
    synthetic_loss_sum = 0.0
    start = time.time()
    model.train()

    progress = tqdm(total=total_steps, desc=f"{args.run_name} mixed training")
    for step_index in range(total_steps):
        optimizer.zero_grad(set_to_none=True)
        if is_synthetic_step(step_index, total_steps, args.synthetic_optimizer_steps):
            update_loss = 0.0
            for _ in range(args.synthetic_accumulation_steps):
                batch = next(synthetic_iterator)
                inputs = batch["input"].to(device, non_blocking=True)
                targets = batch["target"].to(device, non_blocking=True)
                with autocast(enabled=use_amp):
                    pred = model(inputs)
                    base = torch.mean(torch.abs(pred - targets))
                    color = color_loss(pred, targets)
                    tv = residual_tv_loss(pred, inputs, targets)
                    loss = base + args.color_loss_weight * color + args.residual_tv_weight * tv
                    scaled_loss = loss / args.synthetic_accumulation_steps
                scaler.scale(scaled_loss).backward()
                update_loss += float(loss.item()) / args.synthetic_accumulation_steps
            synthetic_steps += 1
            synthetic_loss_sum += update_loss
        else:
            inputs, targets, _meta = next(real_iterator)
            inputs, targets = random_crop_pair(inputs, targets, args.crop_size)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with autocast(enabled=use_amp):
                loss = torch.mean(torch.abs(model(inputs) - targets))
            scaler.scale(loss).backward()
            real_steps += 1
            real_loss_sum += float(loss.item())

        scaler.step(optimizer)
        scaler.update()
        global_step = step_index + 1
        if global_step % args.checkpoint_every == 0:
            save_checkpoint(run_dir / "last.pth", model, optimizer, config, global_step, real_steps, synthetic_steps)
        progress.update(1)
        progress.set_postfix(real=real_steps, synthetic=synthetic_steps)
    progress.close()

    if real_steps != args.real_optimizer_steps or synthetic_steps != args.synthetic_optimizer_steps:
        raise RuntimeError(
            f"Schedule mismatch: real={real_steps}/{args.real_optimizer_steps}, "
            f"synthetic={synthetic_steps}/{args.synthetic_optimizer_steps}"
        )

    save_checkpoint(run_dir / "final.pth", model, optimizer, config, total_steps, real_steps, synthetic_steps)
    real_metrics = evaluate_real(model, real_test_loader, device, args.max_real_eval_samples)
    synthetic_val_metrics = evaluate_synthetic(
        model, synthetic_val_loader, device, args.max_synthetic_val_batches, "evaluate synthetic validation"
    )
    synthetic_test_metrics = evaluate_synthetic(
        model, synthetic_test_loader, device, args.max_synthetic_test_batches, "evaluate synthetic test"
    )
    summary = {
        "run_name": args.run_name,
        "model_key": args.model_key,
        "optimizer_steps": total_steps,
        "real_optimizer_steps": real_steps,
        "synthetic_optimizer_steps": synthetic_steps,
        "synthetic_to_real_update_ratio": ratio,
        "synthetic_fraction_of_all_updates": args.synthetic_optimizer_steps / total_steps,
        "mean_real_train_l1": real_loss_sum / real_steps,
        "mean_synthetic_train_loss": synthetic_loss_sum / synthetic_steps,
        "chamber_holdout": real_metrics,
        "synthetic_validation": synthetic_val_metrics,
        "synthetic_test": synthetic_test_metrics,
        "elapsed_seconds": time.time() - start,
        "timestamp": datetime.now().isoformat(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
