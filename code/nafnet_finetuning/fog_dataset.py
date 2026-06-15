#!/usr/bin/env python3
"""Dataset helpers for synthetic fog removal training."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from synth_fog_tools import FogPreset, PRESETS, synthesize_fog

VALID_SUFFIXES = {".jpg", ".jpeg", ".png"}


def collect_images(root: Path) -> list[Path]:
    paths = [path for path in root.iterdir() if path.is_file() and path.suffix.lower() in VALID_SUFFIXES]
    return sorted(paths)


def split_paths(paths: list[Path], seed: int, train_ratio: float, val_ratio: float) -> dict[str, list[Path]]:
    if not paths:
        raise ValueError("No images found for dataset split.")
    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_paths = shuffled[:train_end]
    val_paths = shuffled[train_end:val_end]
    test_paths = shuffled[val_end:]
    if not train_paths or not val_paths or not test_paths:
        raise ValueError(
            f"Split produced an empty partition: train={len(train_paths)} val={len(val_paths)} test={len(test_paths)}"
        )
    return {"train": train_paths, "val": val_paths, "test": test_paths}


def write_split_manifest(split_map: dict[str, list[Path]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "path"])
        writer.writeheader()
        for split_name, paths in split_map.items():
            for path in paths:
                writer.writerow({"split": split_name, "path": str(path)})


@dataclass(frozen=True)
class SampleRecord:
    clear_path: Path
    split: str


def load_preset_json(path: str | Path) -> FogPreset:
    preset_path = Path(path)
    with preset_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return preset_from_payload(payload, fallback_name=preset_path.stem)


def preset_from_payload(payload: dict[str, object], fallback_name: str = "preset") -> FogPreset:
    return FogPreset(
        name=str(payload.get("name", fallback_name)),
        beta=float(payload["beta"]),
        airlight_rgb=tuple(float(value) for value in payload["airlight_rgb"]),
        bloom_strength=float(payload["bloom_strength"]),
        blur_radius=float(payload["blur_radius"]),
        contrast_gamma=float(payload["contrast_gamma"]),
        saturation_mix=float(payload["saturation_mix"]),
        noise_strength=float(payload.get("noise_strength", 0.0)),
        edge_veil_strength=float(payload["edge_veil_strength"]),
    )


def load_preset_bank_json(path: str | Path) -> list[FogPreset]:
    preset_bank_path = Path(path)
    with preset_bank_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        preset_payloads = payload
    else:
        preset_payloads = payload.get("presets", [])
    presets = [
        preset_from_payload(row, fallback_name=f"{preset_bank_path.stem}_{idx:02d}")
        for idx, row in enumerate(preset_payloads)
    ]
    if not presets:
        raise ValueError(f"No presets found in preset bank: {preset_bank_path}")
    return presets


class SyntheticFogDataset(Dataset):
    def __init__(
        self,
        paths: Iterable[Path],
        split: str,
        fog_level: str = "medium",
        fog_preset: FogPreset | None = None,
        fog_presets: list[FogPreset] | None = None,
        patch_size: int = 384,
        augment: bool = False,
        base_seed: int = 42,
        depth_backend: str = "auto",
        depth_cache_dir: str | Path | None = None,
        allow_model_depth: bool = False,
    ) -> None:
        self.records = [SampleRecord(clear_path=Path(path), split=split) for path in paths]
        self.split = split
        self.patch_size = patch_size
        self.augment = augment
        self.base_seed = base_seed
        self.depth_backend = depth_backend
        self.depth_cache_dir = Path(depth_cache_dir) if depth_cache_dir is not None else None
        self.allow_model_depth = allow_model_depth
        self.fog_level = fog_level
        if fog_presets is not None:
            if not fog_presets:
                raise ValueError("fog_presets cannot be empty.")
            self.fog_presets = list(fog_presets)
        elif fog_preset is not None:
            self.fog_presets = [fog_preset]
        else:
            if fog_level not in PRESETS:
                raise ValueError(f"Unsupported fog level: {fog_level}")
            self.fog_presets = [PRESETS[fog_level]]
        if not self.records:
            raise ValueError(f"No records available for split={split}")

    def __len__(self) -> int:
        return len(self.records)

    def _load_rgb(self, path: Path) -> np.ndarray:
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

    def _crop_pair(self, clear_rgb: np.ndarray, foggy_rgb: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        height, width = clear_rgb.shape[:2]
        crop_h = min(self.patch_size, height)
        crop_w = min(self.patch_size, width)

        if self.split == "train":
            top = int(rng.integers(0, max(1, height - crop_h + 1)))
            left = int(rng.integers(0, max(1, width - crop_w + 1)))
        else:
            top = max(0, (height - crop_h) // 2)
            left = max(0, (width - crop_w) // 2)

        clear_crop = clear_rgb[top : top + crop_h, left : left + crop_w, :]
        foggy_crop = foggy_rgb[top : top + crop_h, left : left + crop_w, :]
        return clear_crop, foggy_crop

    def _apply_augments(self, clear_rgb: np.ndarray, foggy_rgb: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        if not self.augment:
            return clear_rgb, foggy_rgb

        if float(rng.random()) < 0.5:
            clear_rgb = np.flip(clear_rgb, axis=1).copy()
            foggy_rgb = np.flip(foggy_rgb, axis=1).copy()
        if float(rng.random()) < 0.2:
            clear_rgb = np.flip(clear_rgb, axis=0).copy()
            foggy_rgb = np.flip(foggy_rgb, axis=0).copy()
        if float(rng.random()) < 0.25:
            clear_rgb = np.rot90(clear_rgb, k=1).copy()
            foggy_rgb = np.rot90(foggy_rgb, k=1).copy()
        return clear_rgb, foggy_rgb

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        clear_rgb = self._load_rgb(record.clear_path)

        rng = np.random.default_rng(self.base_seed + index * 1009 + (0 if self.split != "train" else np.random.randint(0, 1_000_000)))
        synth_seed = int(rng.integers(0, 1_000_000_000))
        if self.split == "train" and len(self.fog_presets) > 1:
            preset = self.fog_presets[int(rng.integers(0, len(self.fog_presets)))]
        else:
            preset = self.fog_presets[index % len(self.fog_presets)]
        foggy_rgb = synthesize_fog(
            clear_rgb,
            preset,
            seed=synth_seed,
            source_path=record.clear_path,
            depth_backend=self.depth_backend,
            depth_cache_dir=self.depth_cache_dir,
            allow_model_depth=self.allow_model_depth,
        )
        clear_crop, foggy_crop = self._crop_pair(clear_rgb, foggy_rgb, rng)
        clear_crop, foggy_crop = self._apply_augments(clear_crop, foggy_crop, rng)

        input_tensor = torch.from_numpy(np.moveaxis(foggy_crop, -1, 0)).float()
        target_tensor = torch.from_numpy(np.moveaxis(clear_crop, -1, 0)).float()
        sample = {
            "input": input_tensor,
            "target": target_tensor,
            "clear_path": str(record.clear_path),
            "split": self.split,
            "fog_level": preset.name,
        }
        return sample
