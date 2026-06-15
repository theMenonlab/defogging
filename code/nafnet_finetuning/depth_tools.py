#!/usr/bin/env python3
"""Depth-map helpers for model-based fog simulation."""

from __future__ import annotations

import hashlib
import os
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageFilter

MIDAS_MODEL_TYPE = "DPT_Hybrid"
MIDAS_BACKEND_ID = f"midas_{MIDAS_MODEL_TYPE.lower()}_v1"

_MIDAS_STATE: dict[str, object] | None = None
_DEPTH_FAILURES: set[str] = set()


def _normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    if not np.isfinite(arr_min) or not np.isfinite(arr_max) or arr_max - arr_min < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - arr_min) / (arr_max - arr_min), 0.0, 1.0)


def _cache_root(cache_dir: str | Path | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    return Path(__file__).resolve().parent / "depth_cache"


def _cache_path(source_path: Path, cache_dir: str | Path | None, backend_id: str) -> Path:
    resolved = source_path.resolve()
    stat = resolved.stat()
    digest = hashlib.sha1(
        f"{backend_id}|{resolved}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    ).hexdigest()[:16]
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in resolved.stem)
    return _cache_root(cache_dir) / backend_id / f"{safe_stem}_{digest}.npy"


def _load_cached_depth(cache_path: Path) -> np.ndarray | None:
    if not cache_path.exists():
        return None
    arr = np.load(cache_path)
    return np.asarray(arr, dtype=np.float32)


def _save_cached_depth(cache_path: Path, depth_map: np.ndarray) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp.npy")
    np.save(tmp_path, depth_map.astype(np.float16))
    os.replace(tmp_path, cache_path)


def _midas_transform_name(model_type: str) -> str:
    if model_type.startswith("DPT_"):
        return "dpt_transform"
    return "small_transform"


def _get_midas_backend() -> dict[str, object]:
    global _MIDAS_STATE
    if _MIDAS_STATE is not None:
        return _MIDAS_STATE

    import torch

    model = torch.hub.load("isl-org/MiDaS", MIDAS_MODEL_TYPE, trust_repo=True)
    transforms = torch.hub.load("isl-org/MiDaS", "transforms", trust_repo=True)
    transform = getattr(transforms, _midas_transform_name(MIDAS_MODEL_TYPE))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    _MIDAS_STATE = {
        "model": model,
        "transform": transform,
        "device": device,
    }
    return _MIDAS_STATE


def _predict_midas_depth(clear_rgb: np.ndarray) -> np.ndarray:
    import cv2
    import torch

    backend = _get_midas_backend()
    model = backend["model"]
    transform = backend["transform"]
    device = backend["device"]

    rgb_u8 = np.clip(clear_rgb * 255.0, 0, 255).astype(np.uint8)
    input_batch = transform(rgb_u8)
    height, width = clear_rgb.shape[:2]

    with torch.no_grad():
        prediction = model(input_batch.to(device))
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.detach().cpu().numpy().astype(np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth map from MiDaS, got shape={depth.shape}")
    if not np.isfinite(depth).all():
        raise ValueError("MiDaS returned non-finite depth values.")
    return depth


def _orient_and_smooth_depth(
    raw_depth: np.ndarray,
    geometric_prior: np.ndarray,
) -> np.ndarray:
    q_low, q_high = np.quantile(raw_depth, (0.02, 0.98))
    if not np.isfinite(q_low) or not np.isfinite(q_high) or q_high - q_low < 1e-6:
        return geometric_prior.astype(np.float32, copy=True)

    depth = np.clip((raw_depth - q_low) / (q_high - q_low), 0.0, 1.0).astype(np.float32)
    prior = _normalize_map(geometric_prior)

    centered_prior = prior - float(prior.mean())
    corr = float(np.mean((depth - float(depth.mean())) * centered_prior))
    inv_depth = 1.0 - depth
    corr_inv = float(np.mean((inv_depth - float(inv_depth.mean())) * centered_prior))
    if corr_inv > corr:
        depth = inv_depth

    # Keep the model dominant but retain a small geometric stabilizer for sky / horizon regions.
    depth = 0.9 * depth + 0.1 * prior
    depth = np.asarray(
        Image.fromarray(np.clip(depth * 255.0, 0, 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=2.0)
        ),
        dtype=np.float32,
    ) / 255.0
    return np.clip(depth, 0.0, 1.0)


def estimate_depth_like_map(
    clear_rgb: np.ndarray,
    source_path: Path | None,
    backend: str,
    cache_dir: str | Path | None,
    geometric_prior: np.ndarray,
    fallback_depth: np.ndarray,
    allow_model: bool,
) -> np.ndarray:
    backend = backend.lower()
    if backend == "heuristic":
        return fallback_depth.astype(np.float32, copy=True)
    if backend not in {"auto", "midas"}:
        raise ValueError(f"Unsupported depth backend: {backend}")

    cache_path = _cache_path(Path(source_path), cache_dir, MIDAS_BACKEND_ID) if source_path else None
    if cache_path is not None:
        cached = _load_cached_depth(cache_path)
        if cached is not None:
            return cached

    if not allow_model:
        return fallback_depth.astype(np.float32, copy=True)

    try:
        raw_depth = _predict_midas_depth(clear_rgb)
        depth = _orient_and_smooth_depth(raw_depth, geometric_prior)
        if cache_path is not None:
            _save_cached_depth(cache_path, depth)
        return depth
    except Exception as exc:  # pragma: no cover - fallback path
        failure_key = f"{backend}:{type(exc).__name__}:{exc}"
        if failure_key not in _DEPTH_FAILURES:
            warnings.warn(
                f"Falling back to heuristic fog depth because MiDaS depth inference failed: {exc}",
                RuntimeWarning,
            )
            _DEPTH_FAILURES.add(failure_key)
        return fallback_depth.astype(np.float32, copy=True)


def precompute_depth_cache(
    image_paths: Iterable[Path],
    backend: str = "auto",
    cache_dir: str | Path | None = None,
    description: str = "depth cache",
) -> None:
    backend = backend.lower()
    if backend == "heuristic":
        return
    if backend not in {"auto", "midas"}:
        raise ValueError(f"Unsupported depth backend: {backend}")

    paths = [Path(path) for path in image_paths]
    if not paths:
        return

    iterator = paths
    try:
        from tqdm import tqdm

        iterator = tqdm(paths, desc=description)
    except Exception:
        iterator = paths

    from synth_fog_tools import _make_depth_like_map, _make_geometric_depth_prior, load_rgb

    for path in iterator:
        cache_path = _cache_path(path, cache_dir, MIDAS_BACKEND_ID)
        if cache_path.exists():
            continue
        clear_rgb = load_rgb(path)
        geometric_prior = _make_geometric_depth_prior(clear_rgb.shape[0], clear_rgb.shape[1])
        fallback_depth = _make_depth_like_map(clear_rgb.shape[0], clear_rgb.shape[1], seed=0)
        estimate_depth_like_map(
            clear_rgb=clear_rgb,
            source_path=path,
            backend=backend,
            cache_dir=cache_dir,
            geometric_prior=geometric_prior,
            fallback_depth=fallback_depth,
            allow_model=True,
        )
