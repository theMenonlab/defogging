#!/usr/bin/env python3
"""Spatially varying synthetic fog model for tuning and dataset generation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

CODE_DIR = Path(__file__).resolve().parents[1] / "general_code"
sys.path.insert(0, str(CODE_DIR))

from synth_fog_tools import _make_depth_like_map, _make_geometric_depth_prior, load_rgb, save_rgb  # noqa: E402


@dataclass
class SpatialFogPreset:
    name: str = "spatial_gaussian_field_v1"
    beta_mean: float = 1.75
    beta_variation: float = 0.55
    field_scale_px: float = 420.0
    field_octaves: int = 3
    field_contrast: float = 1.20
    vertical_gradient: float = 0.25
    horizon_bias: float = 0.20
    airlight_r: float = 0.72
    airlight_g: float = 0.73
    airlight_b: float = 0.76
    airlight_variation: float = 0.05
    warmth_bias: float = 0.00
    bloom_strength: float = 0.10
    bloom_radius: float = 9.0
    blur_radius: float = 0.9
    blur_fog_coupling: float = 0.22
    saturation_mix: float = 0.24
    contrast_gamma: float = 0.96
    noise_strength: float = 0.008
    edge_veil_strength: float = 0.14
    seed: int = 612


def preset_from_json(path: Path) -> SpatialFogPreset:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    params = payload.get("params", payload)
    valid = {field.name for field in SpatialFogPreset.__dataclass_fields__.values()}
    filtered = {key: value for key, value in params.items() if key in valid}
    return SpatialFogPreset(**filtered)


def save_preset_json(path: Path, preset: SpatialFogPreset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": "spatial_gaussian_field_v1",
        "description": "Spatially varying synthetic fog preset. Transmission uses beta_mean times a smooth random field.",
        "params": asdict(preset),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _resize_noise(noise: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(np.clip(noise * 255.0, 0, 255).astype(np.uint8), mode="L")
    image = image.resize((width, height), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    return (arr - arr.min()) / max(float(arr.max() - arr.min()), 1e-6)


def make_spatial_field(height: int, width: int, preset: SpatialFogPreset) -> np.ndarray:
    rng = np.random.default_rng(int(preset.seed))
    field = np.zeros((height, width), dtype=np.float32)
    total_weight = 0.0
    base_scale = max(24.0, float(preset.field_scale_px))
    octaves = max(1, min(6, int(round(preset.field_octaves))))

    for octave in range(octaves):
        scale = max(16.0, base_scale / (2**octave))
        coarse_h = max(3, int(np.ceil(height / scale)) + 3)
        coarse_w = max(3, int(np.ceil(width / scale)) + 3)
        coarse = rng.random((coarse_h, coarse_w), dtype=np.float32)
        octave_field = _resize_noise(coarse, width, height)
        weight = 1.0 / (1.8**octave)
        field += weight * octave_field
        total_weight += weight

    field = _normalize(field / max(total_weight, 1e-6))
    if preset.field_contrast != 1.0:
        field = np.clip(0.5 + (field - 0.5) * float(preset.field_contrast), 0.0, 1.0)

    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    vertical = 1.0 - yy
    horizon = np.exp(-((yy - 0.38) ** 2) / 0.030).astype(np.float32)
    field = field + float(preset.vertical_gradient) * vertical + float(preset.horizon_bias) * horizon
    return _normalize(field)


def _rgb_to_uint8(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="RGB")


def synthesize_spatial_fog(
    clear_rgb: np.ndarray,
    preset: SpatialFogPreset,
    depth_like: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = clear_rgb.shape[:2]
    field = make_spatial_field(height, width, preset)
    if depth_like is None:
        geometric = _make_geometric_depth_prior(height, width)
        random_depth = _make_depth_like_map(height, width, seed=int(preset.seed) + 101)
        depth_like = _normalize(0.82 * geometric + 0.18 * random_depth)

    beta_map = float(preset.beta_mean) * (
        1.0 + float(preset.beta_variation) * (2.0 * field - 1.0)
    )
    beta_map = np.clip(beta_map, 0.03, 8.0)
    transmission = np.exp(-beta_map * np.clip(depth_like, 0.0, 1.0))[:, :, None]

    rng = np.random.default_rng(int(preset.seed) + 202)
    color_field = np.stack(
        [
            make_spatial_field(height, width, SpatialFogPreset(seed=int(rng.integers(0, 1_000_000)), field_scale_px=preset.field_scale_px * 1.4)),
            make_spatial_field(height, width, SpatialFogPreset(seed=int(rng.integers(0, 1_000_000)), field_scale_px=preset.field_scale_px * 1.4)),
            make_spatial_field(height, width, SpatialFogPreset(seed=int(rng.integers(0, 1_000_000)), field_scale_px=preset.field_scale_px * 1.4)),
        ],
        axis=2,
    )
    base_airlight = np.array(
        [preset.airlight_r, preset.airlight_g, preset.airlight_b], dtype=np.float32
    )[None, None, :]
    warm = np.array([1.0, 0.0, -1.0], dtype=np.float32)[None, None, :] * float(preset.warmth_bias)
    airlight = base_airlight + warm + float(preset.airlight_variation) * (color_field - 0.5)
    airlight = np.clip(airlight, 0.0, 1.0)

    foggy = clear_rgb * transmission + airlight * (1.0 - transmission)

    yy = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    radial = np.sqrt(xx**2 + yy**2)
    edge_veil = np.clip((radial - 0.62) / 0.38, 0.0, 1.0)[:, :, None]
    veil_color = np.clip(base_airlight + np.array([0.01, 0.01, 0.02], dtype=np.float32), 0.0, 1.0)
    foggy = foggy * (1.0 - preset.edge_veil_strength * edge_veil) + veil_color * (
        preset.edge_veil_strength * edge_veil
    )

    if preset.bloom_strength > 0:
        bloom = np.asarray(
            _rgb_to_uint8(foggy).filter(ImageFilter.GaussianBlur(radius=float(preset.bloom_radius))),
            dtype=np.float32,
        ) / 255.0
        foggy = foggy * (1.0 - preset.bloom_strength) + bloom * preset.bloom_strength

    if preset.blur_radius > 0 and preset.blur_fog_coupling > 0:
        blurred = np.asarray(
            _rgb_to_uint8(foggy).filter(ImageFilter.GaussianBlur(radius=float(preset.blur_radius))),
            dtype=np.float32,
        ) / 255.0
        fog_amount = np.clip(1.0 - transmission.mean(axis=2, keepdims=True), 0.0, 1.0)
        blur_mix = np.clip(float(preset.blur_fog_coupling) * fog_amount, 0.0, 0.8)
        foggy = foggy * (1.0 - blur_mix) + blurred * blur_mix

    gray = foggy.mean(axis=2, keepdims=True)
    foggy = foggy * (1.0 - preset.saturation_mix) + gray * preset.saturation_mix
    foggy = np.clip(foggy, 0.0, 1.0) ** max(0.35, float(preset.contrast_gamma))

    if preset.noise_strength > 0:
        noise = rng.normal(0.0, float(preset.noise_strength), size=foggy.shape).astype(np.float32)
        foggy = np.clip(foggy + noise, 0.0, 1.0)

    return np.clip(foggy, 0.0, 1.0), field, np.clip(1.0 - transmission[:, :, 0], 0.0, 1.0)


def make_preview_panel(clear_rgb: np.ndarray, foggy_rgb: np.ndarray, field: np.ndarray, fog_amount: np.ndarray) -> Image.Image:
    height, width = clear_rgb.shape[:2]
    clear = _rgb_to_uint8(clear_rgb)
    foggy = _rgb_to_uint8(foggy_rgb)
    field_rgb = Image.fromarray(np.clip(field * 255.0, 0, 255).astype(np.uint8), mode="L").convert("RGB")
    fog_rgb = Image.fromarray(np.clip(fog_amount * 255.0, 0, 255).astype(np.uint8), mode="L").convert("RGB")
    panel = Image.new("RGB", (width * 4, height), "white")
    for idx, image in enumerate([clear, foggy, field_rgb, fog_rgb]):
        panel.paste(image, (idx * width, 0))
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a spatial synthetic fog preview.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preset-json", type=Path)
    parser.add_argument("--save-preset-json", type=Path)
    args = parser.parse_args()

    preset = preset_from_json(args.preset_json) if args.preset_json else SpatialFogPreset()
    clear_rgb = load_rgb(args.input)
    foggy_rgb, field, fog_amount = synthesize_spatial_fog(clear_rgb, preset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    make_preview_panel(clear_rgb, foggy_rgb, field, fog_amount).save(args.output, quality=94)
    if args.save_preset_json:
        save_preset_json(args.save_preset_json, preset)
    else:
        save_rgb(args.output.with_name(args.output.stem + "_foggy.jpg"), foggy_rgb)


if __name__ == "__main__":
    main()
