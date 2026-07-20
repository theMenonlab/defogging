#!/usr/bin/env python3
"""Precompute a resumable ZoeDepth cache for an image tree.

The output tree mirrors the input tree and replaces image suffixes with
``.npy``. CUDA inference defaults to float16 to reproduce the completed depth
fine-tuning run; any non-finite pixels are repaired before the cache is saved.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


VALID_EXTS = {".jpg", ".jpeg", ".png"}


class ImageDataset(Dataset):
    def __init__(
        self,
        input_root: Path,
        output_root: Path,
        overwrite: bool = False,
        max_images: int | None = None,
    ) -> None:
        self.input_root = input_root
        self.output_root = output_root
        candidates = sorted(
            path
            for path in input_root.rglob("*")
            if path.is_file() and path.suffix.lower() in VALID_EXTS
        )
        if not overwrite:
            candidates = [path for path in candidates if not self.output_path(path).is_file()]
        self.files = candidates[:max_images] if max_images is not None else candidates

    def output_path(self, image_path: Path) -> Path:
        relative = image_path.relative_to(self.input_root)
        return self.output_root / relative.with_suffix(".npy")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[Image.Image, str]:
        path = self.files[index]
        with Image.open(path) as image:
            rgb = image.convert("RGB").copy()
        return rgb, str(path)


def collate_one(batch: list[tuple[Image.Image, str]]) -> tuple[Image.Image, Path]:
    if len(batch) != 1:
        raise ValueError("ZoeDepth preprocessing requires batch size 1 because image sizes vary")
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
    if np.any(depth <= 0):
        positive = depth[depth > 0]
        if not positive.size:
            raise ValueError("ZoeDepth returned no positive pixels")
        depth = depth.copy()
        depth[depth <= 0] = float(np.min(positive))
    return depth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Image directory to scan recursively")
    parser.add_argument("--output", type=Path, required=True, help="Depth-cache root mirroring --input")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=None, help="Optional smoke-test limit")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model", default="Intel/zoedepth-nyu-kitti")
    parser.add_argument(
        "--precision",
        choices=["float16", "float32"],
        default="float16",
        help="Use float16 to reproduce the completed run; float32 changes the metric-depth scale",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 0:
        raise ValueError("--workers must be >= 0")
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = ImageDataset(args.input, args.output, args.overwrite, args.max_images)
    print(f"remaining_images={len(dataset)} input={args.input} output={args.output}", flush=True)
    if not dataset:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    use_float16 = args.precision == "float16" and device.type == "cuda"
    model_dtype = torch.float16 if use_float16 else torch.float32
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForDepthEstimation.from_pretrained(
        args.model,
        torch_dtype=model_dtype,
    ).to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        collate_fn=collate_one,
    )

    with torch.inference_mode():
        for image, image_path in tqdm(loader, desc="ZoeDepth"):
            inputs = processor(images=[image], return_tensors="pt")
            inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
            with torch.autocast(
                device_type=device.type,
                enabled=use_float16,
                dtype=torch.float16,
            ):
                depth = model(**inputs).predicted_depth[0]
            depth_array = sanitize_depth(depth.detach().cpu().numpy())
            output_path = dataset.output_path(image_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = output_path.with_suffix(".npy.partial")
            with temporary.open("wb") as handle:
                np.save(handle, depth_array.astype(np.float16))
            temporary.replace(output_path)


if __name__ == "__main__":
    main()
