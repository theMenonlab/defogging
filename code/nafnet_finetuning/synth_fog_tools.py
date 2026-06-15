#!/usr/bin/env python3
"""Utilities for generating synthetic fog that roughly matches the 3_29_26 dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from PIL import Image, ImageFilter

from depth_tools import estimate_depth_like_map, precompute_depth_cache


@dataclass(frozen=True)
class FogPreset:
    name: str
    beta: float
    airlight_rgb: tuple[float, float, float]
    bloom_strength: float
    blur_radius: float
    contrast_gamma: float
    saturation_mix: float
    noise_strength: float
    edge_veil_strength: float


PRESETS: Dict[str, FogPreset] = {
    "light": FogPreset(
        name="light",
        beta=1.55,
        airlight_rgb=(0.69, 0.71, 0.77),
        bloom_strength=0.08,
        blur_radius=0.6,
        contrast_gamma=0.98,
        saturation_mix=0.24,
        noise_strength=0.010,
        edge_veil_strength=0.18,
    ),
    "medium": FogPreset(
        name="medium",
        beta=2.35,
        airlight_rgb=(0.75, 0.76, 0.81),
        bloom_strength=0.14,
        blur_radius=1.0,
        contrast_gamma=0.92,
        saturation_mix=0.32,
        noise_strength=0.014,
        edge_veil_strength=0.28,
    ),
    "heavy": FogPreset(
        name="heavy",
        beta=3.85,
        airlight_rgb=(0.84, 0.85, 0.89),
        bloom_strength=0.26,
        blur_radius=1.8,
        contrast_gamma=0.84,
        saturation_mix=0.42,
        noise_strength=0.020,
        edge_veil_strength=0.40,
    ),
}


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def save_rgb(path: Path, arr: np.ndarray) -> None:
    clipped = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(clipped).save(path, quality=95)


def compute_stats(arr: np.ndarray) -> Dict[str, float]:
    lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    return {
        "mean_rgb_r": float(arr[:, :, 0].mean()),
        "mean_rgb_g": float(arr[:, :, 1].mean()),
        "mean_rgb_b": float(arr[:, :, 2].mean()),
        "std_rgb_r": float(arr[:, :, 0].std()),
        "std_rgb_g": float(arr[:, :, 1].std()),
        "std_rgb_b": float(arr[:, :, 2].std()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "lum_q10": float(np.quantile(lum, 0.10)),
        "lum_q90": float(np.quantile(lum, 0.90)),
    }


def mean_stats(paths: Iterable[Path]) -> Dict[str, float]:
    stats: List[Dict[str, float]] = []
    for path in paths:
        stats.append(compute_stats(load_rgb(path)))
    keys = stats[0].keys()
    return {key: float(np.mean([row[key] for row in stats])) for key in keys}


def _make_geometric_depth_prior(height: int, width: int) -> np.ndarray:
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]

    horizon = np.clip((0.92 - yy) / 0.92, 0.0, 1.0)
    center = 1.0 - np.sqrt((xx - 0.5) ** 2 / 0.25 + (yy - 0.58) ** 2 / 0.38)
    center = np.clip(center, 0.0, 1.0)
    road_bias = np.clip(1.0 - np.abs(xx - 0.5) / 0.55, 0.0, 1.0) * np.clip(
        (0.95 - yy) / 0.95, 0.0, 1.0
    )
    depth_like = 0.54 * horizon + 0.30 * center + 0.16 * road_bias
    return (depth_like - depth_like.min()) / max(depth_like.max() - depth_like.min(), 1e-6)


def _make_depth_like_map(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coarse = rng.normal(loc=0.0, scale=1.0, size=(height // 8 + 2, width // 8 + 2)).astype(
        np.float32
    )
    coarse = np.kron(coarse, np.ones((8, 8), dtype=np.float32))[:height, :width]
    coarse = (coarse - coarse.min()) / max(coarse.max() - coarse.min(), 1e-6)

    geometric_prior = _make_geometric_depth_prior(height, width)
    depth_like = 0.90 * geometric_prior + 0.10 * coarse
    depth_like = (depth_like - depth_like.min()) / max(depth_like.max() - depth_like.min(), 1e-6)
    return depth_like


def synthesize_fog(
    clear_rgb: np.ndarray,
    preset: FogPreset,
    seed: int = 0,
    source_path: Path | None = None,
    depth_backend: str = "auto",
    depth_cache_dir: str | Path | None = None,
    allow_model_depth: bool = True,
) -> np.ndarray:
    height, width = clear_rgb.shape[:2]
    geometric_prior = _make_geometric_depth_prior(height, width)
    fallback_depth = _make_depth_like_map(height, width, seed=seed)
    depth_like = estimate_depth_like_map(
        clear_rgb=clear_rgb,
        source_path=source_path,
        backend=depth_backend,
        cache_dir=depth_cache_dir,
        geometric_prior=geometric_prior,
        fallback_depth=fallback_depth,
        allow_model=allow_model_depth,
    )
    transmission = np.exp(-preset.beta * depth_like)
    transmission = transmission[:, :, None]

    airlight = np.array(preset.airlight_rgb, dtype=np.float32)[None, None, :]
    foggy = clear_rgb * transmission + airlight * (1.0 - transmission)

    # Small edge veil to mimic optics / windshield contamination visible in the real captures.
    yy = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    radial = np.sqrt(xx**2 + yy**2)
    edge_veil = np.clip((radial - 0.6) / 0.4, 0.0, 1.0)[:, :, None]
    veil_color = np.array((0.70, 0.71, 0.76), dtype=np.float32)[None, None, :]
    foggy = foggy * (1.0 - preset.edge_veil_strength * edge_veil) + veil_color * (
        preset.edge_veil_strength * edge_veil
    )

    pil = Image.fromarray(np.clip(foggy * 255.0, 0, 255).astype(np.uint8))
    bloom = np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=9.0)), dtype=np.float32) / 255.0
    foggy = foggy * (1.0 - preset.bloom_strength) + bloom * preset.bloom_strength

    if preset.blur_radius > 0:
        blur = np.asarray(
            Image.fromarray(np.clip(foggy * 255.0, 0, 255).astype(np.uint8)).filter(
                ImageFilter.GaussianBlur(radius=preset.blur_radius)
            ),
            dtype=np.float32,
        ) / 255.0
        foggy = 0.82 * foggy + 0.18 * blur

    gray = foggy.mean(axis=2, keepdims=True)
    foggy = foggy * (1.0 - preset.saturation_mix) + gray * preset.saturation_mix
    foggy = np.clip(foggy, 0.0, 1.0)
    foggy = foggy ** preset.contrast_gamma

    if preset.noise_strength > 0:
        rng = np.random.default_rng(seed + 17)
        noise = rng.normal(0.0, preset.noise_strength, size=foggy.shape).astype(np.float32)
        foggy = np.clip(foggy + noise, 0.0, 1.0)

    return foggy
