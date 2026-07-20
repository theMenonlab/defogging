# this file loops through the mapillary dataset and runs a depth estimation model on all the images.
# This means that when training the model, it is far faster as the depth has already been precomputed for all the images.
# Inputs:
# Input the mapillary vistas dataset for training. It uses transformers.pipeline to get a depth estimation model
# (Intel/zoedepth-nyu-kitti) from HF/
# and runs it on all the images.
# Outputs:
# Outputs all of the depth map image files to a folder. each depth file is a .npy depth map that is the same name as 
# the original jpg.
# WARNING:
# THE BATCH-SIZE OPTION IS BROKEN! DO NOT PASS A BATCH SIZE!!!!
# call it like:
#python precompute_depth.py \
#    --input /path/to/mapillary \
#    --output /path/to/depth \
#
#   --workers 8
# set workers close to core count
#!/usr/bin/env python3
#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


VALID_EXTS = {".jpg", ".jpeg", ".png"}


class ImageDataset(Dataset):
    def __init__(self, root: Path):
        self.root = Path(root)
        self.files = [
            p for p in self.root.rglob("*")
            if p.suffix.lower() in VALID_EXTS
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert("RGB")
        return img, str(path)


def collate(batch):
    imgs, paths = zip(*batch)
    return list(imgs), list(paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.backends.cudnn.benchmark = True

    processor = AutoImageProcessor.from_pretrained(
        "Intel/zoedepth-nyu-kitti"
    )

    model = AutoModelForDepthEstimation.from_pretrained(
        "Intel/zoedepth-nyu-kitti",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    model.eval()

    ds = ImageDataset(args.input)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate,
    )

    with torch.inference_mode():
        for imgs, paths in tqdm(loader, desc="depth"):
            inputs = processor(images=imgs, return_tensors="pt")
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

            with torch.autocast("cuda", enabled=device == "cuda", dtype=torch.float16):
                outputs = model(**inputs)
                depth = outputs.predicted_depth

            depth = depth.detach().cpu().numpy().astype(np.float16)

            for d, p in zip(depth, paths):
                p = Path(p)
                rel = p.relative_to(args.input)
                out_path = args.output / rel.with_suffix(".npy")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(out_path, d)


if __name__ == "__main__":
    main()
