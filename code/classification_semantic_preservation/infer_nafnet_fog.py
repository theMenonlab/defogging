#!/usr/bin/env python3
"""Run tiled RGB NAFNet inference on real fog images."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from nafnet_arch import NAFNet

VALID_SUFFIXES = {".jpg", ".jpeg", ".png"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IGNORED_BASENAMES = {"synthetic_preview_sheet"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tiled RGB NAFNet inference on foggy photos.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--save-side-by-side", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def collect_images(root: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in root.iterdir()
            if path.is_file()
            and path.suffix.lower() in VALID_SUFFIXES
            and path.stem not in IGNORED_BASENAMES
        ]
    )


class ResidualNAFNetRGB(torch.nn.Module):
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


def build_model(run_config: dict[str, object]) -> torch.nn.Module:
    width = int(run_config["width"])
    middle_blocks = int(run_config["middle_blocks"])
    enc_blocks = list(run_config["enc_blocks"])
    dec_blocks = list(run_config["dec_blocks"])
    if run_config.get("model_type") == "residual_rgb":
        return ResidualNAFNetRGB(width, middle_blocks, enc_blocks, dec_blocks)
    return NAFNet(
        in_channels=3,
        out_channels=3,
        width=width,
        middle_blk_num=middle_blocks,
        enc_blk_nums=enc_blocks,
        dec_blk_nums=dec_blocks,
    )


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def save_rgb(path: Path, array: np.ndarray, quality: int = 95) -> None:
    clipped = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(clipped).save(path, quality=quality)


def iter_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, tile_size - overlap)
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def tiled_inference(model: NAFNet, rgb: np.ndarray, tile_size: int, tile_overlap: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    output = np.zeros((height, width, 3), dtype=np.float32)
    weight = np.zeros((height, width, 1), dtype=np.float32)
    y_starts = iter_starts(height, tile_size, tile_overlap)
    x_starts = iter_starts(width, tile_size, tile_overlap)

    with torch.no_grad():
        for top in y_starts:
            for left in x_starts:
                patch = rgb[top : top + tile_size, left : left + tile_size, :]
                patch_tensor = torch.from_numpy(np.moveaxis(patch, -1, 0)).unsqueeze(0).float().to(DEVICE)
                pred = model(patch_tensor).squeeze(0).cpu().numpy()
                pred = np.moveaxis(np.clip(pred, 0.0, 1.0), 0, -1)

                patch_h, patch_w = patch.shape[:2]
                blend = np.ones((patch_h, patch_w, 1), dtype=np.float32)
                ramp = min(tile_overlap, patch_h // 2, patch_w // 2)
                if ramp > 0:
                    y = np.minimum(np.arange(patch_h), np.arange(patch_h)[::-1]).astype(np.float32)
                    x = np.minimum(np.arange(patch_w), np.arange(patch_w)[::-1]).astype(np.float32)
                    y = np.clip(y / max(1, ramp), 0.0, 1.0)
                    x = np.clip(x / max(1, ramp), 0.0, 1.0)
                    blend = np.minimum.outer(y, x)[:, :, None]
                    blend = np.clip(blend, 1e-3, 1.0)

                output[top : top + patch_h, left : left + patch_w, :] += pred * blend
                weight[top : top + patch_h, left : left + patch_w, :] += blend

    return output / np.clip(weight, 1e-6, None)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.run_config).open("r", encoding="utf-8") as handle:
        run_config = json.load(handle)

    checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model = build_model(run_config).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_paths = collect_images(input_dir)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    manifest_rows = []
    start = time.time()
    for image_path in tqdm(image_paths, desc="real fog inference"):
        output_path = output_dir / image_path.name
        preview_path = output_dir / f"{image_path.stem}_compare.jpg"
        if output_path.exists() and not args.overwrite:
            manifest_rows.append({"input_path": str(image_path), "output_path": str(output_path), "status": "skipped_existing"})
            continue

        rgb = load_rgb(image_path)
        pred = tiled_inference(model, rgb, tile_size=args.tile_size, tile_overlap=args.tile_overlap)
        save_rgb(output_path, pred)
        if args.save_side_by_side:
            compare = np.concatenate([rgb, pred], axis=1)
            save_rgb(preview_path, compare)
        manifest_rows.append({"input_path": str(image_path), "output_path": str(output_path), "status": "written"})

    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": str(DEVICE),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "checkpoint": args.checkpoint,
        "run_config": args.run_config,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "image_count": len(image_paths),
        "elapsed_seconds": time.time() - start,
        "rows": manifest_rows,
    }
    with (output_dir / "inference_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
