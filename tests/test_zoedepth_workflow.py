from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DIR = ROOT / "code" / "synthetic_finetuning"
NAFNET_DIR = ROOT / "code" / "nafnet_finetuning"
sys.path.insert(0, str(NAFNET_DIR))
sys.path.insert(0, str(SYNTHETIC_DIR))

from precompute_depth import ImageDataset, sanitize_depth  # noqa: E402
from spatial_fog_model import SpatialFogPreset, synthesize_spatial_fog  # noqa: E402
from train_spatial_mapillary_nafnet import MapillarySpatialFogDataset  # noqa: E402


def write_rgb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((32, 48, 3), 127, dtype=np.uint8)).save(path)


def test_depth_cache_preserves_nested_relative_paths(tmp_path: Path) -> None:
    input_root = tmp_path / "images"
    output_root = tmp_path / "depth"
    image_path = input_root / "nested" / "sample.jpg"
    write_rgb(image_path)

    dataset = ImageDataset(input_root, output_root)
    assert dataset.output_path(image_path) == output_root / "nested" / "sample.npy"

    expected = dataset.output_path(image_path)
    expected.parent.mkdir(parents=True)
    np.save(expected, np.ones((8, 8), dtype=np.float16))
    assert len(ImageDataset(input_root, output_root)) == 0


def test_training_dataset_uses_same_relative_depth_mapping(tmp_path: Path) -> None:
    image_root = tmp_path / "training" / "images"
    depth_root = tmp_path / "depth" / "training" / "images"
    image_path = image_root / "nested" / "sample.jpg"
    depth_path = depth_root / "nested" / "sample.npy"
    write_rgb(image_path)
    depth_path.parent.mkdir(parents=True)
    np.save(depth_path, np.ones((8, 8), dtype=np.float16))

    dataset = MapillarySpatialFogDataset(
        paths=[image_path],
        split="train",
        preset=SpatialFogPreset(),
        patch_size=16,
        base_seed=1,
        airlight_jitter=0.0,
        beta_mult_min=1.0,
        beta_mult_max=1.0,
        variation_mult_min=1.0,
        variation_mult_max=1.0,
        light_fog_prob=0.0,
        identity_prob=0.0,
        augment=False,
        image_root=image_root,
        depth_root=depth_root,
    )
    assert dataset.samples == [(image_path, depth_path)]


def test_nonfinite_depth_is_sanitized_and_fog_stays_finite() -> None:
    raw = np.array([[1.0, np.nan], [np.inf, -1.0]], dtype=np.float32)
    clean = sanitize_depth(raw)
    assert np.isfinite(clean).all()
    assert (clean > 0).all()

    clear = np.full((8, 8, 3), 0.4, dtype=np.float32)
    extra_depth = np.full((8, 8), 3.0, dtype=np.float32)
    extra_depth[0, 0] = np.nan
    foggy, field, fog_amount = synthesize_spatial_fog(
        clear,
        SpatialFogPreset(beta_mean=0.59, paint_weight=0.7),
        extra_depth=extra_depth,
    )
    assert np.isfinite(foggy).all()
    assert np.isfinite(field).all()
    assert np.isfinite(fog_amount).all()
    assert foggy.min() >= 0.0 and foggy.max() <= 1.0
