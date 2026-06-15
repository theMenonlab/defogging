#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
PAIRED_BENCHMARK_ROOT = PROJECT_ROOT.parent
DEFAULT_LOCAL_FOG_ROOT = PROJECT_ROOT / "VerticalFilter_MediumFog_Redo_3-21-26_aligned"
DEFAULT_LOCAL_GT_ROOT = PROJECT_ROOT / "archive"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT
DEFAULT_RUN6_ROOT = Path(
    PAIRED_BENCHMARK_ROOT / "model_benshmarking" / "run6_colleague_bundle" / "run6"
)
IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
SKIP_MODEL_DIRS = {"phamscope_bassai"}
NATIVE_DEHAZE_RUNS = [
    {
        "model_key": "dehazeformer_fog",
        "model_dir": "phamscope_dehazeformer",
        "train_script": "train_dehazeformer_fog.py",
        "native_rgb": True,
    },
    {
        "model_key": "dcpdn_zhang_fog",
        "model_dir": "phamscope_dcpdn_zhang",
        "train_script": "train_dcpdn_zhang_fog.py",
        "native_rgb": True,
    },
    {
        "model_key": "ancuti_fusion_fog",
        "model_dir": "phamscope_ancuti_fusion",
        "train_script": "train_ancuti_fusion_fog.py",
        "native_rgb": True,
    },
    {
        "model_key": "ffanet_fog",
        "model_dir": "phamscope_ffanet",
        "train_script": "train_ffanet_fog.py",
        "native_rgb": True,
    },
    {
        "model_key": "griddehazenet_fog",
        "model_dir": "phamscope_griddehazenet",
        "train_script": "train_griddehazenet_fog.py",
        "native_rgb": True,
    },
    {
        "model_key": "aodnet_fog",
        "model_dir": "phamscope_aodnet",
        "train_script": "train_aodnet_fog.py",
        "native_rgb": True,
    },
    {
        "model_key": "gcanet_fog",
        "model_dir": "phamscope_gcanet",
        "train_script": "train_gcanet_fog.py",
        "native_rgb": True,
    },
    {
        "model_key": "msbdn_fog",
        "model_dir": "phamscope_msbdn",
        "train_script": "train_msbdn_fog.py",
        "native_rgb": True,
    },
    {
        "model_key": "aecrnet_fog",
        "model_dir": "phamscope_aecrnet",
        "train_script": "train_aecrnet_fog.py",
        "native_rgb": True,
    },
    {
        "model_key": "deanet_fog",
        "model_dir": "phamscope_deanet",
        "train_script": "train_deanet_fog.py",
        "native_rgb": True,
    },
    {
        "model_key": "nafnet_bottleneck1_fog",
        "model_dir": "phamscope_nafnet_bottleneck1",
        "train_script": "train_nafnet_bottleneck1_fog.py",
        "native_rgb": True,
    },
    {
        "model_key": "nafnet_no_sca_fog",
        "model_dir": "phamscope_nafnet_no_sca",
        "train_script": "train_nafnet_no_sca_fog.py",
        "native_rgb": True,
    },
    {
        "model_key": "nafnet_dwt_fog",
        "model_dir": "phamscope_nafnet_dwt",
        "train_script": "train_nafnet_dwt_fog.py",
        "native_rgb": True,
    },
    {
        "model_key": "scope_fog",
        "model_dir": "phamscope_scope_fog",
        "train_script": "train_scope_fog.py",
        "native_rgb": True,
    },
]
NATIVE_DEHAZE_KEYS = {row["model_key"] for row in NATIVE_DEHAZE_RUNS}


class FogRGBPairedDataset(Dataset):
    def __init__(self, fog_root: Path, gt_root: Path, split: str = "all", holdout_every: int = 10) -> None:
        self.fog_root = Path(fog_root)
        self.gt_root = Path(gt_root)
        self.split = split
        self.holdout_every = holdout_every
        self.samples = self._collect_samples()
        if not self.samples:
            raise ValueError(f"No paired RGB samples found: fog={self.fog_root} gt={self.gt_root} split={split}")

    @staticmethod
    def _images(folder: Path) -> list[Path]:
        return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix in IMG_SUFFIXES)

    def _collect_samples(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for category_dir in sorted(p for p in self.fog_root.iterdir() if p.is_dir()):
            gt_category_dir = self.gt_root / category_dir.name
            if not gt_category_dir.is_dir():
                continue
            gt_by_name = {p.name: p for p in self._images(gt_category_dir)}
            category_pairs = []
            for fog_path in self._images(category_dir):
                gt_path = gt_by_name.get(fog_path.name)
                if gt_path is None:
                    continue
                category_pairs.append((fog_path, gt_path))
            for idx, (fog_path, gt_path) in enumerate(category_pairs):
                is_test = idx % self.holdout_every == 0
                if self.split == "train" and is_test:
                    continue
                if self.split == "test" and not is_test:
                    continue
                rows.append(
                    {
                        "category": category_dir.name,
                        "image_name": fog_path.name,
                        "fog_path": str(fog_path),
                        "gt_path": str(gt_path),
                    }
                )
        return rows

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, str]]:
        row = self.samples[idx]
        fog = Image.open(row["fog_path"]).convert("RGB")
        gt = Image.open(row["gt_path"]).convert("RGB")
        if fog.size != (512, 512):
            fog = fog.resize((512, 512), Image.BICUBIC)
        if gt.size != (512, 512):
            gt = gt.resize((512, 512), Image.BICUBIC)
        return to_tensor(fog), to_tensor(gt), row


class SigmoidOutput(nn.Module):
    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.core = core

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.core(x))


class TanhOutput(nn.Module):
    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.core = core

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.core(x) + 1.0) * 0.5


def _child_modules(module: nn.Module) -> list[tuple[nn.Module, str, nn.Module]]:
    children: list[tuple[nn.Module, str, nn.Module]] = []
    for parent in module.modules():
        for name, child in parent.named_children():
            children.append((parent, name, child))
    return children


def _set_child(parent: nn.Module, name: str, child: nn.Module) -> None:
    if isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(name)] = child
    else:
        setattr(parent, name, child)


def _replace_conv(conv: nn.Conv2d, in_channels: int | None = None, out_channels: int | None = None) -> nn.Conv2d:
    new_conv = nn.Conv2d(
        in_channels=in_channels if in_channels is not None else conv.in_channels,
        out_channels=out_channels if out_channels is not None else conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    )
    return new_conv


def make_direct_rgb_model(core: nn.Module, model_key: str, output_wrapper: type[nn.Module] = SigmoidOutput) -> nn.Module:
    """Convert old 1->120 spectral cores into direct 3->3 RGB models.

    The previous wrapper learned a 3->1 input projection, ran the original
    spectral core, then learned a 120->3 output projection. For RGB fog removal
    that hides color from the core. This keeps the core architecture but rewires
    its boundary convolutions so the model itself sees RGB and predicts RGB.
    """

    first_replaced = False
    for parent, name, child in _child_modules(core):
        if isinstance(child, nn.Conv2d) and child.in_channels == 1:
            _set_child(parent, name, _replace_conv(child, in_channels=3))
            first_replaced = True
            break

    last_replaced = False
    for parent, name, child in reversed(_child_modules(core)):
        if isinstance(child, nn.Conv2d) and child.out_channels == 120:
            _set_child(parent, name, _replace_conv(child, out_channels=3))
            last_replaced = True
            break

    if not first_replaced or not last_replaced:
        raise RuntimeError(
            f"{model_key}: could not rewire old spectral core to direct RGB "
            f"(first_in_1={first_replaced}, last_out_120={last_replaced})"
        )
    return output_wrapper(core)


@dataclass(frozen=True)
class ModelRun:
    model_key: str
    model_dir: str
    train_script: str
    native_rgb: bool = False


def import_module_from_path(name: str, path: Path):
    stale = [
        "dataset",
        "early_stopping",
        "models",
        "models.networks",
        "network_module",
        "PixelUnShuffle",
        "realmask_common",
        "specat_simple",
        "common",
        "reggan_components",
    ]
    for mod_name in stale:
        sys.modules.pop(mod_name, None)
    parent = str(path.parent)
    run6 = str(path.parent.parent)
    for entry in [parent, run6]:
        if entry not in sys.path:
            sys.path.insert(0, entry)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_model_runs(run6_root: Path) -> list[ModelRun]:
    collect_path = run6_root / "collect_fc_results.py"
    module = import_module_from_path("fog_benchmark_collect_fc_results", collect_path)
    runs: list[ModelRun] = []
    for row in module.MODEL_RUNS:
        if row["model_dir"] in SKIP_MODEL_DIRS:
            continue
        runs.append(ModelRun(row["model_key"], row["model_dir"], row["train_script"].split()[0]))
    for row in NATIVE_DEHAZE_RUNS:
        train_path = run6_root / row["model_dir"] / row["train_script"]
        if train_path.exists():
            runs.append(
                ModelRun(
                    row["model_key"],
                    row["model_dir"],
                    row["train_script"],
                    bool(row["native_rgb"]),
                )
            )
    return runs


def build_model_for_run(run6_root: Path, run: ModelRun) -> nn.Module:
    model_dir = run6_root / run.model_dir
    train_path = model_dir / run.train_script
    if run.model_key == "gmsr_fc":
        module = import_module_from_path(f"fog_benchmark_{run.model_key}_gmsr_simple", model_dir / "gmsr_simple.py")
        return SigmoidOutput(module.GMSR_Simple(inp_channels=3, out_channels=3))
    if run.model_key == "hrnet_fc":
        module = import_module_from_path(f"fog_benchmark_{run.model_key}_hrnet", model_dir / "hrnet_model.py",)
        opt = module.Options()
        opt.in_channels = 3
        opt.out_channels = 3
        return SigmoidOutput(module.SGN(opt))
    if run.model_key == "pix2pix_fc":
        module = import_module_from_path(f"fog_benchmark_{run.model_key}_networks", model_dir / "models" / "networks.py")
        return TanhOutput(
            module.define_G(
                input_nc=3,
                output_nc=3,
                ngf=64,
                netG="unet_128",
                norm="batch",
                use_dropout=False,
                init_type="normal",
                init_gain=0.02,
            )
        )
    if run.model_key == "sr3_fc":
        module = import_module_from_path(f"fog_benchmark_{run.model_key}_sr3_simple", model_dir / "sr3_simple.py")
        return SigmoidOutput(module.SR3_Simple(input_channels=3, output_channels=3))
    if run.model_key == "reggan_fc":
        module = import_module_from_path(f"fog_benchmark_{run.model_key}_common", model_dir / "common.py")
        return make_direct_rgb_model(module.build_generator(), run.model_key, TanhOutput)
    if run.model_key.startswith("specat"):
        specat_module = import_module_from_path(f"fog_benchmark_{run.model_key}_specat", model_dir / "specat_simple.py")
        stage = 2 if "_s2" in run.model_key else 1
        return SigmoidOutput(specat_module.SPECAT_Simple(input_channels=3, output_channels=3, stage=stage))
    module = import_module_from_path(f"fog_benchmark_{run.model_key}", train_path)
    if not hasattr(module, "build_model"):
        raise AttributeError(f"{train_path} does not expose build_model()")
    model = module.build_model()
    if run.native_rgb or bool(getattr(module, "NATIVE_RGB", False)):
        return model
    return make_direct_rgb_model(model, run.model_key)


def dataset_report(fog_root: Path, gt_root: Path, output_path: Path, holdout_every: int) -> dict[str, Any]:
    rows = []
    total_pairs = 0
    identical_pairs = 0
    problems: list[str] = []
    for category_dir in sorted(p for p in fog_root.iterdir() if p.is_dir()):
        gt_category_dir = gt_root / category_dir.name
        fog_files = FogRGBPairedDataset._images(category_dir)
        gt_files = FogRGBPairedDataset._images(gt_category_dir) if gt_category_dir.is_dir() else []
        gt_by_name = {p.name: p for p in gt_files}
        matched = [p for p in fog_files if p.name in gt_by_name]
        category_identical = sum(1 for p in matched if p.read_bytes() == gt_by_name[p.name].read_bytes())
        total_pairs += len(matched)
        identical_pairs += category_identical
        if len(matched) != len(fog_files):
            problems.append(f"{category_dir.name}: {len(fog_files) - len(matched)} fog images missing exact GT match")
        if matched and category_identical == len(matched):
            problems.append(f"{category_dir.name}: all {category_identical} fog/GT pairs are byte-identical")
        rows.append(
            {
                "category": category_dir.name,
                "fog_images": len(fog_files),
                "gt_images": len(gt_files),
                "exact_pairs": len(matched),
                "identical_pairs": category_identical,
                "train_pairs": len([p for i, p in enumerate(matched) if i % holdout_every != 0]),
                "test_pairs": len([p for i, p in enumerate(matched) if i % holdout_every == 0]),
            }
        )
    if total_pairs and identical_pairs == total_pairs:
        problems.append(f"all {identical_pairs} exact fog/GT pairs are byte-identical")
    report = {
        "timestamp": datetime.now().isoformat(),
        "fog_root": str(fog_root),
        "gt_root": str(gt_root),
        "holdout_every": holdout_every,
        "total_exact_pairs": total_pairs,
        "identical_pairs": identical_pairs,
        "categories": rows,
        "problems": problems,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def tensor_to_uint8(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (arr * 255.0).round().clip(0, 255).astype(np.uint8)


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.clip(pred, 0.0, 1.0)
    target = np.clip(target, 0.0, 1.0)
    return {
        "mae": float(np.mean(np.abs(pred - target))),
        "mse": float(np.mean((pred - target) ** 2)),
        "psnr": float(peak_signal_noise_ratio(target, pred, data_range=1.0)),
        "ssim": float(structural_similarity(target, pred, channel_axis=2, data_range=1.0)),
    }


def validate_models(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_root = Path(args.output_root)
    report_path = output_root / "validation" / "dataset_report.json"
    dataset_summary = dataset_report(Path(args.fog_root), Path(args.gt_root), report_path, args.holdout_every)
    if dataset_summary["problems"]:
        raise SystemExit("Dataset validation failed:\n- " + "\n- ".join(dataset_summary["problems"]))

    dataset = FogRGBPairedDataset(Path(args.fog_root), Path(args.gt_root), split="train", holdout_every=args.holdout_every)
    subset = Subset(dataset, list(range(min(args.max_samples, len(dataset)))))
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)
    sample_inputs, sample_targets, _ = next(iter(loader))
    if args.patch_size and args.patch_size < sample_inputs.shape[-1]:
        sample_inputs = sample_inputs[:, :, : args.patch_size, : args.patch_size]
        sample_targets = sample_targets[:, :, : args.patch_size, : args.patch_size]
    sample_inputs = sample_inputs.to(device)
    sample_targets = sample_targets.to(device)

    rows = []
    for run in load_model_runs(Path(args.run6_root)):
        row: dict[str, Any] = {"model_key": run.model_key, "status": "pending", "seconds": None, "error": ""}
        start = time.time()
        try:
            model = build_model_for_run(Path(args.run6_root), run).to(device)
            model.eval()
            with torch.no_grad():
                output = model(sample_inputs)
                loss = torch.mean(torch.abs(output - sample_targets)).item()
            row.update(
                {
                    "status": "ok",
                    "seconds": time.time() - start,
                    "output_shape": list(output.shape),
                    "smoke_l1": loss,
                    "parameters": sum(p.numel() for p in model.parameters()),
                }
            )
        except Exception as exc:
            row.update({"status": "failed", "seconds": time.time() - start, "error": repr(exc)})
        rows.append(row)
        print(f"{row['model_key']}: {row['status']} {row.get('error', '')}", flush=True)

    out_path = output_root / "validation" / "model_validation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"timestamp": datetime.now().isoformat(), "device": str(device), "rows": rows}, indent=2), encoding="utf-8")
    failures = [r for r in rows if r["status"] != "ok"]
    if failures:
        raise SystemExit(f"{len(failures)} model validations failed; see {out_path}")


def train_one(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_root = Path(args.output_root)
    run = {r.model_key: r for r in load_model_runs(Path(args.run6_root))}[args.model_key]
    model_dir = output_root / "checkpoints" / args.model_key
    results_dir = output_root / "results" / args.model_key
    logs_dir = output_root / "logs" / args.model_key
    for directory in [model_dir, results_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    train_dataset = FogRGBPairedDataset(Path(args.fog_root), Path(args.gt_root), split="train", holdout_every=args.holdout_every)
    if args.max_train_samples:
        train_dataset = Subset(train_dataset, list(range(min(args.max_train_samples, len(train_dataset)))))
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    model = build_model_for_run(Path(args.run6_root), run).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.L1Loss()
    use_amp = bool(args.use_amp and device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)
    start_epoch = 0
    resume_from = Path(args.resume_from) if args.resume_from else None
    if args.auto_resume and resume_from is None:
        checkpoints = sorted(model_dir.glob("epoch_*.pth"))
        if checkpoints:
            resume_from = checkpoints[-1]
    if resume_from is not None and resume_from.exists():
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        start_epoch = int(checkpoint.get("epoch", 0))
        print(f"Resuming {args.model_key} from {resume_from} at epoch {start_epoch}", flush=True)

    log = {
        "timestamp": datetime.now().isoformat(),
        "model_key": args.model_key,
        "fog_root": args.fog_root,
        "gt_root": args.gt_root,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "use_amp": use_amp,
        "train_crop_size": args.train_crop_size,
        "learning_rate": args.learning_rate,
        "grad_clip": args.grad_clip,
        "resume_from": str(resume_from) if resume_from else "",
        "start_epoch": start_epoch,
        "train_samples": len(train_dataset),
        "parameters": sum(p.numel() for p in model.parameters()),
        "epoch_rows": [],
    }
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total = 0.0
        batches = 0
        nonfinite_batches = 0
        for batch_idx, (fog, gt, _meta) in enumerate(tqdm(loader, desc=f"{args.model_key} epoch {epoch + 1}/{args.epochs}")):
            fog = fog.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            if args.train_crop_size and args.train_crop_size < fog.shape[-1]:
                crop = int(args.train_crop_size)
                max_y = fog.shape[-2] - crop
                max_x = fog.shape[-1] - crop
                top = int(torch.randint(0, max_y + 1, (1,), device=fog.device).item()) if max_y > 0 else 0
                left = int(torch.randint(0, max_x + 1, (1,), device=fog.device).item()) if max_x > 0 else 0
                fog = fog[:, :, top : top + crop, left : left + crop]
                gt = gt[:, :, top : top + crop, left : left + crop]
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                pred = model(fog)
                loss = criterion(pred, gt)
            if not torch.isfinite(loss):
                nonfinite_batches += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
            batches += 1
            if args.max_batches and batch_idx + 1 >= args.max_batches:
                break
        mean_loss = total / batches if batches else math.nan
        log["epoch_rows"].append({"epoch": epoch + 1, "train_l1": mean_loss, "batches": batches, "nonfinite_batches": nonfinite_batches})
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "epoch": epoch + 1,
                "train_l1": mean_loss,
            },
            model_dir / f"epoch_{epoch + 1:03d}.pth",
        )
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "args": vars(args), "epoch": args.epochs}, model_dir / "final.pth")
    (logs_dir / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    evaluate_model(model, Path(args.fog_root), Path(args.gt_root), results_dir, args.holdout_every, args.max_eval_samples, device)


def evaluate_model(model: nn.Module, fog_root: Path, gt_root: Path, results_dir: Path, holdout_every: int, max_eval_samples: int | None, device: torch.device) -> None:
    dataset = FogRGBPairedDataset(fog_root, gt_root, split="test", holdout_every=holdout_every)
    if max_eval_samples:
        dataset = Subset(dataset, list(range(min(max_eval_samples, len(dataset)))))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    image_dir = results_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    model.eval()
    with torch.no_grad():
        for fog, gt, meta in tqdm(loader, desc="eval"):
            fog = fog.to(device)
            pred = model(fog).squeeze(0).cpu().clamp(0, 1)
            gt_img = gt.squeeze(0)
            pred_np = pred.permute(1, 2, 0).numpy()
            gt_np = gt_img.permute(1, 2, 0).numpy()
            row = {k: v[0] for k, v in meta.items()}
            row.update(compute_metrics(pred_np, gt_np))
            rows.append(row)
            stem = f"{row['category']}_{Path(row['image_name']).stem}"
            Image.fromarray(tensor_to_uint8(pred)).save(image_dir / f"{stem}_pred.png")
            Image.fromarray(tensor_to_uint8(fog.squeeze(0).cpu())).save(image_dir / f"{stem}_input.png")
            Image.fromarray(tensor_to_uint8(gt_img)).save(image_dir / f"{stem}_target.png")
    if rows:
        with (results_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "samples": len(rows),
        "mean_mae": float(np.mean([r["mae"] for r in rows])) if rows else math.nan,
        "mean_mse": float(np.mean([r["mse"] for r in rows])) if rows else math.nan,
        "mean_psnr": float(np.mean([r["psnr"] for r in rows])) if rows else math.nan,
        "mean_ssim": float(np.mean([r["ssim"] for r in rows])) if rows else math.nan,
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_chpc_scripts(args: argparse.Namespace) -> None:
    out = Path(args.output_root)
    slurm_dir = out / "chpc" / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    chpc_root = Path(args.chpc_root)
    run6_chpc = chpc_root / "run6_colleague_bundle" / "run6"
    mail_line = f"#SBATCH --mail-user={args.mail_user}\n" if args.mail_user else ""
    model_keys = [r.model_key for r in load_model_runs(Path(args.run6_root))]
    runner = out / "chpc" / "run_one_fog_model.sh"
    runner.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
MODEL_KEY="${{1:?model key required}}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXTRA_ARGS=()
case "$MODEL_KEY" in
  dehazeformer_fog|dcpdn_zhang_fog|ancuti_fusion_fog|ffanet_fog|griddehazenet_fog|aodnet_fog|gcanet_fog|msbdn_fog|aecrnet_fog|deanet_fog|nafnet_bottleneck1_fog|nafnet_no_sca_fog|nafnet_dwt_fog|scope_fog)
    EXTRA_ARGS+=(--train-crop-size 256)
    ;;
  hat_fc)
    EXTRA_ARGS+=(--train-crop-size 128)
    ;;
  mirnet_fc|restormer_fc|swin2sr_fc|swinir_fc)
    EXTRA_ARGS+=(--train-crop-size 256)
    ;;
esac
set +u
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  if conda env list | awk '{{print $1}}' | grep -qx hsrmamba; then
    conda activate hsrmamba
  elif conda env list | awk '{{print $1}}' | grep -qx pix2pix; then
    conda activate pix2pix
  else
    echo "Neither hsrmamba nor pix2pix conda env is available." >&2
    conda info --envs >&2
    exit 1
  fi
else
  source activate pix2pix
fi
set -u
cd "$PROJECT_ROOT"
python fog_rgb_benchmark.py train-one \\
  --model-key "$MODEL_KEY" \\
  --fog-root "{args.chpc_fog_root}" \\
  --gt-root "{args.chpc_gt_root}" \\
  --output-root "{args.chpc_root}" \\
  --run6-root "{run6_chpc}" \\
  --epochs {args.epochs} \\
  --batch-size {args.batch_size} \\
  --num-workers 8 \\
  "${{EXTRA_ARGS[@]}}"
""",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    for key in model_keys:
        path = slurm_dir / f"{key}.slurm"
        path.write_text(
            f"""#!/bin/bash
#SBATCH --account={args.account}
#SBATCH --partition={args.partition}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --job-name=fogbench_{key}
#SBATCH --output={args.chpc_root}/logs/slurm/fogbench_{key}_%j.log
#SBATCH --mail-type=FAIL,BEGIN,END
{mail_line}

mkdir -p "{args.chpc_root}/logs/slurm" "{args.chpc_root}/checkpoints" "{args.chpc_root}/results"
cd "{args.chpc_root}/chpc"
bash ./run_one_fog_model.sh "{key}"
""",
            encoding="utf-8",
        )
    submit = out / "chpc" / "submit_all_fog_benchmark_jobs.sh"
    submit.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd \"$(dirname \"$0\")/slurm\"\nfor job in *.slurm; do echo \"sbatch $job\"; sbatch \"$job\"; done\n",
        encoding="utf-8",
    )
    submit.chmod(0o755)
    upload = out / "chpc" / "upload_to_chpc.sh"
    upload.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
DEST="${{REMOTE_DEST:?set REMOTE_DEST to user@host:/path}}"
rsync -av \\
  --exclude '__pycache__' \\
  --exclude 'validation' \\
  --exclude 'checkpoints' \\
  --exclude 'results' \\
  --exclude 'logs' \\
  --exclude '*.pth' \\
  "{out}/" "$DEST/"
rsync -av --exclude '__pycache__' "{Path(args.run6_root).parent}/" "$DEST/run6_colleague_bundle/"
""",
        encoding="utf-8",
    )
    upload.chmod(0o755)
    (out / "chpc" / "MODEL_KEYS.txt").write_text("\n".join(model_keys) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--fog-root", default=str(DEFAULT_LOCAL_FOG_ROOT))
    common.add_argument("--gt-root", default=str(DEFAULT_LOCAL_GT_ROOT))
    common.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    common.add_argument("--run6-root", default=str(DEFAULT_RUN6_ROOT))
    common.add_argument("--holdout-every", type=int, default=10)
    common.add_argument("--device", default="")

    p = sub.add_parser("validate", parents=[common])
    p.add_argument("--max-samples", type=int, default=1)
    p.add_argument("--patch-size", type=int, default=128)
    p.set_defaults(func=validate_models)

    p = sub.add_parser("train-one", parents=[common])
    p.add_argument("--model-key", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--train-crop-size", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--resume-from", default="")
    p.add_argument("--auto-resume", action="store_true")
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.set_defaults(func=train_one)

    p = sub.add_parser("write-chpc", parents=[common])
    p.add_argument("--chpc-root", default="/path/to/20260523_fog_benchmarking")
    p.add_argument("--chpc-fog-root", default="/path/to/20260523_fog_benchmarking/VerticalFilter_MediumFog_Redo_3-21-26_aligned")
    p.add_argument("--chpc-gt-root", default="/path/to/20260523_fog_benchmarking/archive")
    p.add_argument("--account", default="gpu-account")
    p.add_argument("--partition", default="gpu-partition")
    p.add_argument("--mail-user", default="")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1)
    p.set_defaults(func=write_chpc_scripts)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
