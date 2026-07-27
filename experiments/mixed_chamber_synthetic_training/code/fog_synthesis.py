#!/usr/bin/env python3
"""Paper-current randomized fog plus Andrew's ZoeDepth extension."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class FogPreset:
    name: str = "next_warm_low_bloom_guard"
    beta_mean: float = 2.58
    beta_variation: float = 0.94
    field_scale_px: float = 924.0
    field_octaves: int = 6
    field_contrast: float = 1.2
    vertical_gradient: float = 0.25
    horizon_bias: float = 0.2
    airlight_r: float = 0.72
    airlight_g: float = 0.715
    airlight_b: float = 0.7
    airlight_variation: float = 0.045
    warmth_bias: float = 0.0
    bloom_strength: float = 0.18
    bloom_radius: float = 8.0
    blur_radius: float = 0.9
    blur_fog_coupling: float = 0.22
    saturation_mix: float = 0.14
    contrast_gamma: float = 1.0
    noise_strength: float = 0.008
    edge_veil_strength: float = 0.1
    paint_weight: float = 1.0
    seed: int = 2673


def load_preset(path: Path) -> FogPreset:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload = payload.get("params", payload)
    allowed = {f.name for f in fields(FogPreset)}
    return FogPreset(**{k: v for k, v in payload.items() if k in allowed})


def jitter_preset(
    preset: FogPreset,
    rng: np.random.Generator,
    beta_mult_min: float = 0.55,
    beta_mult_max: float = 1.40,
    variation_mult_min: float = 0.60,
    variation_mult_max: float = 1.15,
    airlight_jitter: float = 0.03,
    light_fog_prob: float = 0.30,
    identity_prob: float = 0.10,
) -> tuple[FogPreset, bool]:
    if float(rng.random()) < identity_prob:
        return preset, True
    beta_mult = float(rng.uniform(beta_mult_min, beta_mult_max))
    variation_mult = float(rng.uniform(variation_mult_min, variation_mult_max))
    if float(rng.random()) < light_fog_prob:
        beta_mult *= float(rng.uniform(0.12, 0.45))
        variation_mult *= float(rng.uniform(0.25, 0.70))
    jitter = rng.uniform(-airlight_jitter, airlight_jitter, size=3)
    return replace(
        preset,
        seed=int(rng.integers(0, 10_000_000)),
        beta_mean=float(preset.beta_mean * beta_mult),
        beta_variation=float(preset.beta_variation * variation_mult),
        airlight_r=float(np.clip(preset.airlight_r + jitter[0], 0.0, 1.0)),
        airlight_g=float(np.clip(preset.airlight_g + jitter[1], 0.0, 1.0)),
        airlight_b=float(np.clip(preset.airlight_b + jitter[2], 0.0, 1.0)),
    ), False


def normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        raise ValueError("Depth map contains no finite values")
    replacement = float(np.median(arr[finite]))
    arr = np.nan_to_num(arr, nan=replacement, posinf=replacement, neginf=replacement)
    lo, hi = np.percentile(arr, [1.0, 99.0])
    if float(hi - lo) < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / float(hi - lo), 0.0, 1.0).astype(np.float32)


def _resize_noise(noise: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(np.clip(noise * 255.0, 0, 255).astype(np.uint8), mode="L")
    image = image.resize((width, height), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def make_spatial_field(height: int, width: int, preset: FogPreset) -> np.ndarray:
    rng = np.random.default_rng(int(preset.seed))
    field = np.zeros((height, width), dtype=np.float32)
    total_weight = 0.0
    for octave in range(max(1, min(6, int(round(preset.field_octaves))))):
        scale = max(16.0, max(24.0, preset.field_scale_px) / (2**octave))
        coarse_h = max(3, int(np.ceil(height / scale)) + 3)
        coarse_w = max(3, int(np.ceil(width / scale)) + 3)
        weight = 1.0 / (1.8**octave)
        field += weight * _resize_noise(rng.random((coarse_h, coarse_w), dtype=np.float32), width, height)
        total_weight += weight
    field = normalize_map(field / max(total_weight, 1e-6))
    field = np.clip(0.5 + (field - 0.5) * float(preset.field_contrast), 0.0, 1.0)
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    field += float(preset.vertical_gradient) * (1.0 - yy)
    field += float(preset.horizon_bias) * np.exp(-((yy - 0.38) ** 2) / 0.030)
    return normalize_map(field)


def geometric_depth(height: int, width: int) -> np.ndarray:
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    horizon = np.clip((0.92 - yy) / 0.92, 0.0, 1.0)
    center = np.clip(1.0 - np.sqrt((xx - 0.5) ** 2 / 0.25 + (yy - 0.58) ** 2 / 0.38), 0.0, 1.0)
    road = np.clip(1.0 - np.abs(xx - 0.5) / 0.55, 0.0, 1.0) * np.clip((0.95 - yy) / 0.95, 0.0, 1.0)
    return normalize_map(0.54 * horizon + 0.30 * center + 0.16 * road)


def randomized_depth(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coarse = rng.normal(size=(height // 8 + 2, width // 8 + 2)).astype(np.float32)
    coarse = np.kron(coarse, np.ones((8, 8), dtype=np.float32))[:height, :width]
    return normalize_map(0.90 * geometric_depth(height, width) + 0.10 * normalize_map(coarse))


def _rgb_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="RGB")


def synthesize_fog(
    clear_rgb: np.ndarray,
    preset: FogPreset,
    mode: str,
    zoe_depth: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Render fog.

    ``randomized`` reproduces the paper-current geometric/random depth-like map.
    ``andrew_depth`` retains that base map and applies Andrew's depth-sensitive
    beta modulation to raw metric depth. Non-finite values fall back to the
    randomized spatial beta and the final beta map is bounded to [0.03, 8].
    """
    if mode not in {"randomized", "andrew_depth"}:
        raise ValueError(f"Unknown fog mode: {mode}")
    height, width = clear_rgb.shape[:2]
    field = make_spatial_field(height, width, preset)
    base_depth = normalize_map(
        0.82 * geometric_depth(height, width)
        + 0.18 * randomized_depth(height, width, int(preset.seed) + 101)
    )
    beta_map = float(preset.beta_mean) * (1.0 + float(preset.beta_variation) * (2.0 * field - 1.0))
    if mode == "andrew_depth":
        if zoe_depth is None:
            raise ValueError("andrew_depth mode requires a ZoeDepth map")
        valid_depth = np.isfinite(zoe_depth) & (zoe_depth > 0)
        beta_map[valid_depth] = float(preset.beta_mean) * (
            1.0 + float(preset.paint_weight) * zoe_depth[valid_depth]
        )
    beta_map = np.clip(beta_map, 0.03, 8.0)
    transmission = np.exp(-beta_map * base_depth)[:, :, None]

    rng = np.random.default_rng(int(preset.seed) + 202)
    color_fields = []
    for _ in range(3):
        child = replace(
            preset,
            seed=int(rng.integers(0, 1_000_000)),
            field_scale_px=preset.field_scale_px * 1.4,
        )
        color_fields.append(make_spatial_field(height, width, child))
    color_field = np.stack(color_fields, axis=2)
    base_airlight = np.array([preset.airlight_r, preset.airlight_g, preset.airlight_b], dtype=np.float32)[None, None, :]
    warm = np.array([1.0, 0.0, -1.0], dtype=np.float32)[None, None, :] * float(preset.warmth_bias)
    airlight = np.clip(base_airlight + warm + float(preset.airlight_variation) * (color_field - 0.5), 0.0, 1.0)
    foggy = clear_rgb * transmission + airlight * (1.0 - transmission)

    yy = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    radial = np.sqrt(xx**2 + yy**2)
    edge = np.clip((radial - 0.62) / 0.38, 0.0, 1.0)[:, :, None]
    veil = np.clip(base_airlight + np.array([0.01, 0.01, 0.02], dtype=np.float32), 0.0, 1.0)
    foggy = foggy * (1.0 - preset.edge_veil_strength * edge) + veil * (preset.edge_veil_strength * edge)

    if preset.bloom_strength > 0:
        bloom = np.asarray(_rgb_image(foggy).filter(ImageFilter.GaussianBlur(preset.bloom_radius)), dtype=np.float32) / 255.0
        foggy = foggy * (1.0 - preset.bloom_strength) + bloom * preset.bloom_strength
    if preset.blur_radius > 0 and preset.blur_fog_coupling > 0:
        blurred = np.asarray(_rgb_image(foggy).filter(ImageFilter.GaussianBlur(preset.blur_radius)), dtype=np.float32) / 255.0
        amount = np.clip(1.0 - transmission.mean(axis=2, keepdims=True), 0.0, 1.0)
        mix = np.clip(float(preset.blur_fog_coupling) * amount, 0.0, 0.8)
        foggy = foggy * (1.0 - mix) + blurred * mix
    gray = foggy.mean(axis=2, keepdims=True)
    foggy = foggy * (1.0 - preset.saturation_mix) + gray * preset.saturation_mix
    foggy = np.clip(foggy, 0.0, 1.0) ** max(0.35, float(preset.contrast_gamma))
    if preset.noise_strength > 0:
        foggy = np.clip(foggy + rng.normal(0.0, preset.noise_strength, size=foggy.shape), 0.0, 1.0)
    stats = {
        "beta_min": float(beta_map.min()),
        "beta_mean": float(beta_map.mean()),
        "beta_max": float(beta_map.max()),
        "transmission_min": float(transmission.min()),
        "transmission_mean": float(transmission.mean()),
        "transmission_max": float(transmission.max()),
    }
    return np.asarray(foggy, dtype=np.float32), stats
