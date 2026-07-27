#!/usr/bin/env python3
"""Precompute raw ZoeDepth arrays for one Mapillary split, safely resumable."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


VALID_SUFFIXES = {".jpg", ".jpeg", ".png"}


class Images(Dataset):
    def __init__(self, input_root: Path, output_root: Path) -> None:
        self.input_root = input_root
        self.output_root = output_root
        candidates = sorted(p for p in input_root.rglob("*") if p.is_file() and p.suffix.lower() in VALID_SUFFIXES)
        self.files = [p for p in candidates if not self.output_path(p).is_file()]

    def output_path(self, path: Path) -> Path:
        return self.output_root / path.relative_to(self.input_root).with_suffix(".npy")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[Image.Image, str]:
        path = self.files[index]
        with Image.open(path) as image:
            rgb = image.convert("RGB").copy()
        return rgb, str(path)


def collate_one(batch: list[tuple[Image.Image, str]]) -> tuple[Image.Image, Path]:
    if len(batch) != 1:
        raise ValueError("This script intentionally requires batch size 1 because Mapillary image sizes vary")
    image, path = batch[0]
    return image, Path(path)


def sanitize_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth)
    if not finite.any():
        raise ValueError("ZoeDepth returned no finite pixels")
    if not finite.all():
        depth = depth.copy()
        depth[~finite] = float(np.median(depth[finite]))
    return depth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default="Intel/zoedepth-nyu-kitti")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = Images(args.input, args.output)
    print(f"remaining={len(dataset)} input={args.input} output={args.output}", flush=True)
    if not dataset:
        return
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
        collate_fn=collate_one,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForDepthEstimation.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    with torch.inference_mode():
        for image, path in tqdm(loader, desc="ZoeDepth"):
            inputs = processor(images=[image], return_tensors="pt")
            inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
                depth = model(**inputs).predicted_depth[0]
            array = sanitize_depth(depth.detach().cpu().numpy()).astype(np.float16)
            out_path = dataset.output_path(path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = out_path.with_suffix(".npy.partial")
            with temporary.open("wb") as handle:
                np.save(handle, array)
            temporary.replace(out_path)


if __name__ == "__main__":
    main()
