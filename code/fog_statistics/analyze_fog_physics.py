#!/usr/bin/env python3
"""Fog physics report for the paired fog-imager benchmark.

The analysis is intentionally conservative. The camera/display/fog chamber data
are not radiometrically calibrated, so transmission and optical depth are
reported as apparent quantities estimated from paired sRGB images.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy import ndimage, stats


CHAMBER_DEPTH_M = 0.114
CHAMBER_TRANSVERSE_MM = (133, 114)
RGB_WAVELENGTH_NM = np.array([620.0, 540.0, 460.0], dtype=np.float64)
TUNED_AIRLIGHT = np.array([0.620, 0.6406603229737036, 0.680], dtype=np.float64)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class PairRecord:
    split: str
    category: str
    image_name: str
    fog_path: Path
    clear_path: Path


def read_rgb(path: Path, size: int | None = None) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def dark_channel_haze_index(rgb: np.ndarray, patch_size: int = 15) -> float:
    dark = np.min(rgb, axis=2)
    dark = ndimage.minimum_filter(dark, size=patch_size, mode="nearest")
    return float(np.mean(dark))


def proxy_fog_metrics(rgb: np.ndarray) -> dict[str, float]:
    lum = luminance(rgb)
    gy, gx = np.gradient(lum)
    grad = np.sqrt(gx * gx + gy * gy)
    lap = ndimage.laplace(lum)
    local_mean = ndimage.uniform_filter(lum, size=31, mode="reflect")
    local_sq_mean = ndimage.uniform_filter(lum * lum, size=31, mode="reflect")
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean * local_mean, 0.0))
    dark = np.min(rgb, axis=2)
    dark = ndimage.minimum_filter(dark, size=15, mode="nearest")
    hsv_s = np.asarray(Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8)).convert("HSV"))[..., 1]
    hist, _ = np.histogram(lum.reshape(-1), bins=64, range=(0.0, 1.0))
    prob = hist.astype(np.float64) / max(float(hist.sum()), 1.0)
    prob = prob[prob > 0]
    lum_q = np.quantile(lum, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    dark_q = np.quantile(dark, [0.10, 0.50, 0.90, 0.95])
    grad_q = np.quantile(grad, [0.50, 0.90, 0.95, 0.99])
    sat_q = np.quantile(hsv_s.astype(np.float32) / 255.0, [0.50, 0.90])
    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    colorfulness = math.sqrt(float(np.var(rg) + np.var(yb))) + 0.3 * math.sqrt(
        float(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    )
    freq, power = radial_power(lum, bins=48)
    total_power = float(np.sum(power)) + 1e-12
    power_prob = power / total_power
    power_prob_nonzero = power_prob[power_prob > 0]
    low_mask = freq < 0.05
    mid_mask = (freq >= 0.05) & (freq < 0.15)
    high_mask = freq >= 0.15
    lum_mean = float(np.mean(lum))
    lum_std = float(np.std(lum))
    return {
        "dark_channel_haze_index": dark_channel_haze_index(rgb),
        "dark_channel_std": float(np.std(dark)),
        "dark_channel_q10": float(dark_q[0]),
        "dark_channel_q50": float(dark_q[1]),
        "dark_channel_q90": float(dark_q[2]),
        "dark_channel_q95": float(dark_q[3]),
        "rms_contrast": lum_std / max(lum_mean, 1e-9),
        "luminance_std": lum_std,
        "luminance_q01": float(lum_q[0]),
        "luminance_q05": float(lum_q[1]),
        "luminance_q10": float(lum_q[2]),
        "luminance_q25": float(lum_q[3]),
        "luminance_q50": float(lum_q[4]),
        "luminance_q75": float(lum_q[5]),
        "luminance_q90": float(lum_q[6]),
        "luminance_q95": float(lum_q[7]),
        "luminance_q99": float(lum_q[8]),
        "luminance_iqr": float(lum_q[5] - lum_q[3]),
        "luminance_p90_p10_range": float(lum_q[6] - lum_q[2]),
        "luminance_p95_p05_range": float(lum_q[7] - lum_q[1]),
        "michelson_p95_p05": float((lum_q[7] - lum_q[1]) / max(lum_q[7] + lum_q[1], 1e-9)),
        "gradient_magnitude_mean": float(np.mean(grad)),
        "gradient_magnitude_std": float(np.std(grad)),
        "gradient_magnitude_q50": float(grad_q[0]),
        "gradient_magnitude_q90": float(grad_q[1]),
        "gradient_magnitude_q95": float(grad_q[2]),
        "gradient_magnitude_q99": float(grad_q[3]),
        "edge_density_grad_gt_0p01": float(np.mean(grad > 0.01)),
        "edge_density_grad_gt_0p02": float(np.mean(grad > 0.02)),
        "tenengrad": float(np.mean(grad * grad)),
        "brenner_sharpness": float(
            0.5
            * (
                np.mean((lum[:, 2:] - lum[:, :-2]) ** 2)
                + np.mean((lum[2:, :] - lum[:-2, :]) ** 2)
            )
        ),
        "mean_intensity": lum_mean,
        "luminance_entropy_bits": float(-np.sum(prob * np.log2(prob))),
        "laplacian_variance": float(np.var(lap)),
        "saturation_mean": float(np.mean(hsv_s) / 255.0),
        "saturation_std": float(np.std(hsv_s.astype(np.float32) / 255.0)),
        "saturation_q50": float(sat_q[0]),
        "saturation_q90": float(sat_q[1]),
        "colorfulness": float(colorfulness),
        "rgb_channel_std_mean": float(np.mean(np.std(rgb, axis=(0, 1)))),
        "rgb_channel_range_mean": float(np.mean(np.quantile(rgb, 0.95, axis=(0, 1)) - np.quantile(rgb, 0.05, axis=(0, 1)))),
        "local_contrast_mean": float(np.mean(local_std)),
        "local_contrast_std": float(np.std(local_std)),
        "local_contrast_q90": float(np.quantile(local_std, 0.90)),
        "fft_low_power_fraction": float(np.sum(power[low_mask]) / total_power),
        "fft_mid_power_fraction": float(np.sum(power[mid_mask]) / total_power),
        "fft_high_power_fraction": float(np.sum(power[high_mask]) / total_power),
        "fft_high_to_low_power": float(np.sum(power[high_mask]) / max(np.sum(power[low_mask]), 1e-12)),
        "fft_spectral_centroid": float(np.sum(freq * power) / total_power),
        "fft_spectral_entropy": float(-np.sum(power_prob_nonzero * np.log2(power_prob_nonzero))),
    }


def estimate_airlight(fog_rgb: np.ndarray, patch_size: int = 15, top_fraction: float = 0.001) -> np.ndarray:
    dark = np.min(fog_rgb, axis=2)
    dark = ndimage.minimum_filter(dark, size=patch_size, mode="nearest")
    flat_dark = dark.reshape(-1)
    flat_rgb = fog_rgb.reshape(-1, 3)
    n_top = max(1, int(round(top_fraction * flat_dark.size)))
    candidate_idx = np.argpartition(flat_dark, -n_top)[-n_top:]
    candidate_lum = luminance(flat_rgb[candidate_idx])
    return flat_rgb[candidate_idx[np.argmax(candidate_lum)]].astype(np.float32)


def apparent_transmission(fog_rgb: np.ndarray, clear_rgb: np.ndarray, airlight: np.ndarray) -> np.ndarray:
    air = airlight.reshape(1, 1, 3)
    denom = clear_rgb - air
    numer = fog_rgb - air
    valid = np.abs(denom) > 0.04
    with np.errstate(divide="ignore", invalid="ignore"):
        t_channels = np.where(valid, numer / denom, np.nan)
    valid_count = np.sum(np.isfinite(t_channels), axis=2)
    t = np.nanmedian(t_channels, axis=2)
    t[valid_count < 2] = np.nan
    # Transmission is physically bounded. Values outside the range are usually
    # caused by sRGB nonlinearity, saturated pixels, or an unstable airlight
    # denominator, so exclude them from apparent optical-depth summaries.
    t[(t <= 0.0) | (t > 1.0)] = np.nan
    return t.astype(np.float32)


def summarize_t(t: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(t)
    if not np.any(finite):
        return {
            "t_valid_fraction": 0.0,
            "t_mean": np.nan,
            "t_std": np.nan,
            "t_q10": np.nan,
            "t_q50": np.nan,
            "t_q90": np.nan,
            "tau_mean": np.nan,
            "beta_app_m_inv": np.nan,
        }
    vals = np.clip(t[finite], 1e-3, 1.0)
    tau = -np.log(vals)
    return {
        "t_valid_fraction": float(finite.mean()),
        "t_mean": float(np.nanmean(vals)),
        "t_std": float(np.nanstd(vals)),
        "t_q10": float(np.nanquantile(vals, 0.10)),
        "t_q50": float(np.nanquantile(vals, 0.50)),
        "t_q90": float(np.nanquantile(vals, 0.90)),
        "tau_mean": float(np.mean(tau)),
        "beta_app_m_inv": float(np.mean(tau) / CHAMBER_DEPTH_M),
    }


def summarize_input_t(t: np.ndarray, prefix: str) -> dict[str, float]:
    finite = np.isfinite(t)
    if not np.any(finite):
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_q10": np.nan,
            f"{prefix}_q50": np.nan,
            f"{prefix}_q90": np.nan,
        }
    vals = t[finite]
    return {
        f"{prefix}_mean": float(np.mean(vals)),
        f"{prefix}_std": float(np.std(vals)),
        f"{prefix}_q10": float(np.quantile(vals, 0.10)),
        f"{prefix}_q50": float(np.quantile(vals, 0.50)),
        f"{prefix}_q90": float(np.quantile(vals, 0.90)),
    }


def dark_channel_prior_transmission(
    fog_rgb: np.ndarray,
    patch_size: int = 15,
    omega: float = 0.95,
    top_fraction: float = 0.001,
    t_floor: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    airlight = estimate_airlight(fog_rgb, patch_size=patch_size, top_fraction=top_fraction)
    normalized = fog_rgb / np.maximum(airlight.reshape(1, 1, 3), 1e-6)
    dark = np.min(normalized, axis=2)
    dark = ndimage.minimum_filter(dark, size=patch_size, mode="nearest")
    t = 1.0 - omega * dark
    t = np.clip(t, t_floor, 1.0)
    return t.astype(np.float32), airlight


def radial_power(gray: np.ndarray, bins: int = 96) -> tuple[np.ndarray, np.ndarray]:
    h, w = gray.shape
    ywin = np.hanning(h)[:, None]
    xwin = np.hanning(w)[None, :]
    centered = gray - float(np.mean(gray))
    spectrum = np.fft.fftshift(np.fft.fft2(centered * ywin * xwin))
    power = np.abs(spectrum) ** 2
    yy, xx = np.indices((h, w), dtype=np.float32)
    radius = np.sqrt((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2)
    rmax = min(h, w) / 2.0
    bin_edges = np.linspace(0.0, rmax, bins + 1)
    which = np.digitize(radius.reshape(-1), bin_edges) - 1
    sums = np.bincount(which, weights=power.reshape(-1), minlength=bins)
    counts = np.bincount(which, minlength=bins)
    radial = sums[:bins] / np.maximum(counts[:bins], 1)
    freq = 0.5 * (bin_edges[:-1] + bin_edges[1:]) / min(h, w)
    return freq, radial


def mutual_information_bits(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    hist, _, _ = np.histogram2d(a.reshape(-1), b.reshape(-1), bins=bins, range=[[0, 1], [0, 1]])
    pxy = hist / max(float(hist.sum()), 1.0)
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    denom = px @ py
    valid = (pxy > 0) & (denom > 0)
    return float(np.sum(pxy[valid] * np.log2(pxy[valid] / denom[valid])))


def resolve_pairs(metrics_csv: Path, fog_root: Path, clear_root: Path) -> list[PairRecord]:
    metrics = pd.read_csv(metrics_csv)
    test_keys = {(r.category, r.image_name) for r in metrics.itertuples(index=False)}
    categories = sorted(metrics["category"].unique())
    records: list[PairRecord] = []
    for category in categories:
        fog_dir = fog_root / category
        clear_dir = clear_root / category
        clear_by_stem = {
            p.stem: p
            for p in clear_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        }
        fog_paths = sorted(
            p for p in fog_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        for fog_path in fog_paths:
            clear_path = clear_by_stem.get(fog_path.stem)
            if clear_path is None:
                continue
            split = "test" if (category, fog_path.name) in test_keys else "train"
            if split != "test" and (category, f"{fog_path.stem}.jpg") in test_keys:
                split = "test"
            if split != "test" and (category, f"{fog_path.stem}.jpeg") in test_keys:
                split = "test"
            if split != "test" and (category, f"{fog_path.stem}.png") in test_keys:
                split = "test"
            records.append(PairRecord(split, category, fog_path.name, fog_path, clear_path))
    return records


def load_metrics(metrics_csv: Path) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_csv).copy()
    metrics = metrics.rename(
        columns={
            "mae": "nafnet_mae",
            "mse": "nafnet_mse",
            "psnr": "nafnet_psnr",
            "ssim": "nafnet_ssim",
        }
    )
    return metrics


def analyze_pairs(records: list[PairRecord], analysis_size: int, out_dir: Path) -> pd.DataFrame:
    rows = []
    mean_t_acc = np.zeros((analysis_size, analysis_size), dtype=np.float64)
    mean_t_count = np.zeros((analysis_size, analysis_size), dtype=np.float64)

    for idx, rec in enumerate(records, start=1):
        fog = read_rgb(rec.fog_path, size=analysis_size)
        clear = read_rgb(rec.clear_path, size=analysis_size)
        air = estimate_airlight(fog)
        t = apparent_transmission(fog, clear, air)
        dcp_t, _ = dark_channel_prior_transmission(fog)
        finite = np.isfinite(t)
        mean_t_acc[finite] += t[finite]
        mean_t_count[finite] += 1.0
        summ = summarize_t(t)
        dcp_summ = summarize_input_t(dcp_t, "dcp_t")
        fog_proxy = proxy_fog_metrics(fog)
        clear_proxy = proxy_fog_metrics(clear)
        clear_l = luminance(clear)
        fog_l = luminance(fog)
        mi = mutual_information_bits(clear_l, fog_l)
        mse = float(np.mean((clear - fog) ** 2))
        input_psnr = float(10.0 * math.log10(1.0 / max(mse, 1e-12)))
        rows.append(
            {
                "split": rec.split,
                "category": rec.category,
                "image_name": rec.image_name,
                "fog_path": str(rec.fog_path),
                "clear_path": str(rec.clear_path),
                "airlight_r": float(air[0]),
                "airlight_g": float(air[1]),
                "airlight_b": float(air[2]),
                "input_psnr": input_psnr,
                "mi_bits": mi,
                **{f"fog_{k}": v for k, v in fog_proxy.items()},
                **{f"clear_{k}": v for k, v in clear_proxy.items()},
                "dark_channel_delta": fog_proxy["dark_channel_haze_index"] - clear_proxy["dark_channel_haze_index"],
                "rms_contrast_ratio": fog_proxy["rms_contrast"] / max(clear_proxy["rms_contrast"], 1e-9),
                "luminance_std_ratio": fog_proxy["luminance_std"] / max(clear_proxy["luminance_std"], 1e-9),
                "gradient_ratio": fog_proxy["gradient_magnitude_mean"] / max(clear_proxy["gradient_magnitude_mean"], 1e-9),
                "mean_intensity_shift": fog_proxy["mean_intensity"] - clear_proxy["mean_intensity"],
                **summ,
                **dcp_summ,
            }
        )
        if idx % 500 == 0:
            print(f"processed {idx}/{len(records)} paired images", flush=True)

    mean_t_map = np.divide(mean_t_acc, mean_t_count, out=np.full_like(mean_t_acc, np.nan), where=mean_t_count > 0)
    np.save(out_dir / "mean_apparent_transmission_map.npy", mean_t_map.astype(np.float32))
    return pd.DataFrame(rows)


def analyze_psd(
    paired_df: pd.DataFrame,
    outdoor_fog_dir: Path,
    clear_dashcam_dir: Path,
    analysis_size: int,
) -> pd.DataFrame:
    rows = []

    def accumulate(label: str, paths: Iterable[Path], max_n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        acc = None
        freq = None
        count = 0
        path_list = [p for p in list(paths) if p.suffix.lower() in IMAGE_EXTS]
        for path in path_list[:max_n]:
            gray = luminance(read_rgb(path, size=analysis_size))
            freq_i, psd = radial_power(gray)
            if acc is None:
                acc = np.zeros_like(psd, dtype=np.float64)
                freq = freq_i
            acc += psd
            count += 1
        if acc is None or freq is None:
            raise RuntimeError(f"No PSD images for {label}")
        return freq, acc / max(count, 1)

    test_df = paired_df[paired_df["split"] == "test"]
    chamber_fog_paths = [Path(p) for p in test_df["fog_path"]]
    chamber_clear_paths = [Path(p) for p in test_df["clear_path"]]
    freq, chamber_fog = accumulate("chamber fog test", chamber_fog_paths)
    _, chamber_clear = accumulate("chamber clear test", chamber_clear_paths)
    outdoor_paths = sorted(p for p in outdoor_fog_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    dashcam_paths = sorted(p for p in clear_dashcam_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    _, outdoor_fog = accumulate("outdoor fog proxy", outdoor_paths)
    _, dashcam_clear = accumulate("clear dashcam proxy", dashcam_paths)

    eps = 1e-12
    for i, f in enumerate(freq):
        rows.append(
            {
                "frequency_cycles_per_pixel": float(f),
                "chamber_fog_power": float(chamber_fog[i]),
                "chamber_clear_power": float(chamber_clear[i]),
                "chamber_fog_to_clear_ratio": float(chamber_fog[i] / max(chamber_clear[i], eps)),
                "outdoor_fog_power": float(outdoor_fog[i]),
                "clear_dashcam_power": float(dashcam_clear[i]),
                "outdoor_fog_to_clear_dashcam_proxy_ratio": float(outdoor_fog[i] / max(dashcam_clear[i], eps)),
            }
        )
    return pd.DataFrame(rows)


def summarize_proxy_metrics(paired_df: pd.DataFrame, outdoor_fog_dir: Path, clear_dashcam_dir: Path, analysis_size: int) -> pd.DataFrame:
    rows = []

    test = paired_df[paired_df["split"] == "test"]
    paired_ratios = []
    fog_metrics = []
    clear_metrics = []
    for rec in test.itertuples(index=False):
        fog = proxy_fog_metrics(read_rgb(Path(rec.fog_path), size=analysis_size))
        clear = proxy_fog_metrics(read_rgb(Path(rec.clear_path), size=analysis_size))
        fog_metrics.append(fog)
        clear_metrics.append(clear)
        paired_ratios.append(
            {
                "dark_channel_delta": fog["dark_channel_haze_index"] - clear["dark_channel_haze_index"],
                "rms_contrast_ratio": fog["rms_contrast"] / max(clear["rms_contrast"], 1e-9),
                "luminance_std_ratio": fog["luminance_std"] / max(clear["luminance_std"], 1e-9),
                "gradient_ratio": fog["gradient_magnitude_mean"] / max(clear["gradient_magnitude_mean"], 1e-9),
                "mean_intensity_shift": fog["mean_intensity"] - clear["mean_intensity"],
            }
        )

    def mean_metric(dicts: list[dict[str, float]], key: str) -> float:
        return float(np.mean([d[key] for d in dicts])) if dicts else float("nan")

    def std_metric(dicts: list[dict[str, float]], key: str) -> float:
        return float(np.std([d[key] for d in dicts], ddof=1)) if len(dicts) > 1 else float("nan")

    rows.append(
        {
            "condition": "Benchmark chamber",
            "domain": "chamber",
            "state": "fog",
            "samples": len(fog_metrics),
            "pairing": "paired clear archive",
            "reference": "paired clear archive target",
            "dark_channel_haze_index_mean": mean_metric(fog_metrics, "dark_channel_haze_index"),
            "dark_channel_haze_index_std": std_metric(fog_metrics, "dark_channel_haze_index"),
            "rms_contrast_mean": mean_metric(fog_metrics, "rms_contrast"),
            "rms_contrast_std": std_metric(fog_metrics, "rms_contrast"),
            "gradient_magnitude_mean_mean": mean_metric(fog_metrics, "gradient_magnitude_mean"),
            "gradient_magnitude_mean_std": std_metric(fog_metrics, "gradient_magnitude_mean"),
            "mean_intensity_mean": mean_metric(fog_metrics, "mean_intensity"),
            "dark_channel_delta_vs_reference_mean": mean_metric(paired_ratios, "dark_channel_delta"),
            "rms_contrast_ratio_vs_reference_mean": mean_metric(paired_ratios, "rms_contrast_ratio"),
            "luminance_std_ratio_vs_reference_mean": mean_metric(paired_ratios, "luminance_std_ratio"),
            "gradient_ratio_vs_reference_mean": mean_metric(paired_ratios, "gradient_ratio"),
            "mean_intensity_shift_vs_reference_mean": mean_metric(paired_ratios, "mean_intensity_shift"),
        }
    )

    outdoor_clear_paths = sorted(p for p in clear_dashcam_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    outdoor_fog_paths = sorted(p for p in outdoor_fog_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    outdoor_clear = [proxy_fog_metrics(read_rgb(p, size=analysis_size)) for p in outdoor_clear_paths]
    outdoor_fog = [proxy_fog_metrics(read_rgb(p, size=analysis_size)) for p in outdoor_fog_paths]
    clear_means = {k: mean_metric(outdoor_clear, k) for k in ["dark_channel_haze_index", "rms_contrast", "luminance_std", "gradient_magnitude_mean", "mean_intensity"]}
    fog_means = {k: mean_metric(outdoor_fog, k) for k in ["dark_channel_haze_index", "rms_contrast", "luminance_std", "gradient_magnitude_mean", "mean_intensity"]}

    rows.append(
        {
            "condition": "Outdoor clear",
            "domain": "outdoor",
            "state": "clear",
            "samples": len(outdoor_clear),
            "pairing": "independent outdoor group",
            "reference": "self baseline",
            "dark_channel_haze_index_mean": clear_means["dark_channel_haze_index"],
            "dark_channel_haze_index_std": std_metric(outdoor_clear, "dark_channel_haze_index"),
            "rms_contrast_mean": clear_means["rms_contrast"],
            "rms_contrast_std": std_metric(outdoor_clear, "rms_contrast"),
            "gradient_magnitude_mean_mean": clear_means["gradient_magnitude_mean"],
            "gradient_magnitude_mean_std": std_metric(outdoor_clear, "gradient_magnitude_mean"),
            "mean_intensity_mean": clear_means["mean_intensity"],
            "dark_channel_delta_vs_reference_mean": 0.0,
            "rms_contrast_ratio_vs_reference_mean": 1.0,
            "luminance_std_ratio_vs_reference_mean": 1.0,
            "gradient_ratio_vs_reference_mean": 1.0,
            "mean_intensity_shift_vs_reference_mean": 0.0,
        }
    )
    rows.append(
        {
            "condition": "Outdoor fog",
            "domain": "outdoor",
            "state": "fog",
            "samples": len(outdoor_fog),
            "pairing": "independent outdoor group",
            "reference": "outdoor clear group mean",
            "dark_channel_haze_index_mean": fog_means["dark_channel_haze_index"],
            "dark_channel_haze_index_std": std_metric(outdoor_fog, "dark_channel_haze_index"),
            "rms_contrast_mean": fog_means["rms_contrast"],
            "rms_contrast_std": std_metric(outdoor_fog, "rms_contrast"),
            "gradient_magnitude_mean_mean": fog_means["gradient_magnitude_mean"],
            "gradient_magnitude_mean_std": std_metric(outdoor_fog, "gradient_magnitude_mean"),
            "mean_intensity_mean": fog_means["mean_intensity"],
            "dark_channel_delta_vs_reference_mean": fog_means["dark_channel_haze_index"] - clear_means["dark_channel_haze_index"],
            "rms_contrast_ratio_vs_reference_mean": fog_means["rms_contrast"] / max(clear_means["rms_contrast"], 1e-9),
            "luminance_std_ratio_vs_reference_mean": fog_means["luminance_std"] / max(clear_means["luminance_std"], 1e-9),
            "gradient_ratio_vs_reference_mean": fog_means["gradient_magnitude_mean"] / max(clear_means["gradient_magnitude_mean"], 1e-9),
            "mean_intensity_shift_vs_reference_mean": fog_means["mean_intensity"] - clear_means["mean_intensity"],
        }
    )
    return pd.DataFrame(rows)


def correlation_rows(test_physics: pd.DataFrame) -> pd.DataFrame:
    label_overrides = {
        "fog_dark_channel_haze_index": "Foggy dark-channel haze index",
        "fog_rms_contrast": "Foggy RMS contrast",
        "fog_luminance_std": "Foggy luminance standard deviation",
        "fog_gradient_magnitude_mean": "Foggy mean gradient magnitude",
        "fog_mean_intensity": "Foggy mean intensity",
        "fog_luminance_entropy_bits": "Foggy luminance entropy",
        "fog_laplacian_variance": "Foggy Laplacian variance",
        "fog_saturation_mean": "Foggy mean saturation",
        "dcp_t_mean": "DCP mean estimated transmission",
        "dcp_t_q10": "DCP 10th-percentile estimated transmission",
        "dcp_t_q50": "DCP median estimated transmission",
        "dcp_t_std": "DCP estimated-transmission standard deviation",
    }

    def label_for(column: str) -> str:
        if column in label_overrides:
            return label_overrides[column]
        if column.startswith("fog_"):
            return "Foggy " + column.removeprefix("fog_").replace("_", " ")
        if column.startswith("dcp_"):
            return "DCP " + column.removeprefix("dcp_").replace("_", " ")
        return column.replace("_", " ")

    input_columns = [
        c
        for c in test_physics.columns
        if (c.startswith("fog_") or c.startswith("dcp_"))
        and pd.api.types.is_numeric_dtype(test_physics[c])
    ]
    candidates = [("input_only", label_for(c), c) for c in sorted(input_columns)]
    candidates += [
        ("paired_diagnostic", "Input PSNR against clear target", "input_psnr"),
        ("paired_diagnostic", "Clear/fog mutual information", "mi_bits"),
        ("paired_diagnostic", "Mean apparent transmission", "t_mean"),
        ("paired_diagnostic", "Median apparent transmission", "t_q50"),
        ("paired_diagnostic", "Apparent optical depth", "tau_mean"),
        ("paired_diagnostic", "Apparent extinction coefficient", "beta_app_m_inv"),
        ("paired_diagnostic", "Dark-channel increase over clear target", "dark_channel_delta"),
        ("paired_diagnostic", "RMS contrast ratio to clear target", "rms_contrast_ratio"),
        ("paired_diagnostic", "Gradient ratio to clear target", "gradient_ratio"),
        ("paired_diagnostic", "Mean intensity shift from clear target", "mean_intensity_shift"),
    ]
    rows = []
    y = test_physics["nafnet_psnr"]
    y_centered = y - test_physics.groupby("category")["nafnet_psnr"].transform("mean")
    for metric_group, label, column in candidates:
        if column not in test_physics.columns:
            continue
        x = test_physics[column]
        valid = np.isfinite(x) & np.isfinite(y)
        if int(valid.sum()) < 4 or float(np.nanstd(x[valid])) == 0.0:
            continue
        pearson = stats.pearsonr(x[valid], y[valid])
        spearman = stats.spearmanr(x[valid], y[valid])
        x_centered = x - test_physics.groupby("category")[column].transform("mean")
        valid_centered = np.isfinite(x_centered) & np.isfinite(y_centered)
        if int(valid_centered.sum()) >= 4 and float(np.nanstd(x_centered[valid_centered])) > 0.0:
            pearson_within = stats.pearsonr(x_centered[valid_centered], y_centered[valid_centered])
            spearman_within = stats.spearmanr(x_centered[valid_centered], y_centered[valid_centered])
            pearson_within_r = float(pearson_within.statistic)
            pearson_within_p = float(pearson_within.pvalue)
            spearman_within_rho = float(spearman_within.statistic)
            spearman_within_p = float(spearman_within.pvalue)
        else:
            pearson_within_r = pearson_within_p = spearman_within_rho = spearman_within_p = np.nan
        rows.append(
            {
                "metric_group": metric_group,
                "metric_label": label,
                "column": column,
                "n": int(valid.sum()),
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
                "category_centered_pearson_r": pearson_within_r,
                "category_centered_pearson_p": pearson_within_p,
                "category_centered_spearman_rho": spearman_within_rho,
                "category_centered_spearman_p": spearman_within_p,
                "abs_spearman_rho": abs(float(spearman.statistic)),
                "abs_category_centered_spearman_rho": abs(spearman_within_rho) if not pd.isna(spearman_within_rho) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["metric_group", "abs_spearman_rho"], ascending=[True, False]).reset_index(drop=True)


def write_csvs(
    paired_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    psd_df: pd.DataFrame,
    proxy_df: pd.DataFrame,
    out_dir: Path,
) -> dict[str, Path]:
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    paired_path = tables / "apparent_transmission_per_image.csv"
    paired_df.to_csv(paired_path, index=False)

    test_physics = paired_df[paired_df["split"] == "test"].merge(
        metrics_df[["category", "image_name", "nafnet_psnr", "nafnet_ssim", "nafnet_mae", "nafnet_mse"]],
        on=["category", "image_name"],
        how="left",
    )
    test_path = tables / "test_physics_plus_nafnet_metrics.csv"
    test_physics.to_csv(test_path, index=False)

    corr = correlation_rows(test_physics)
    corr_path = tables / "fog_statistic_psnr_correlations.csv"
    corr.to_csv(corr_path, index=False)

    cat = (
        test_physics.groupby("category")
        .agg(
            n=("image_name", "count"),
            input_psnr_mean=("input_psnr", "mean"),
            nafnet_psnr_mean=("nafnet_psnr", "mean"),
            nafnet_psnr_std=("nafnet_psnr", "std"),
            nafnet_ssim_mean=("nafnet_ssim", "mean"),
            t_mean=("t_mean", "mean"),
            t_std_mean=("t_std", "mean"),
            tau_mean=("tau_mean", "mean"),
            beta_app_m_inv=("beta_app_m_inv", "mean"),
            mi_bits_mean=("mi_bits", "mean"),
        )
        .reset_index()
    )
    cat_path = tables / "per_category_physics_and_restoration.csv"
    cat.to_csv(cat_path, index=False)

    summary_rows = []
    for split, sub in paired_df.groupby("split"):
        summary_rows.append(
            {
                "split": split,
                "n": len(sub),
                "t_mean": sub["t_mean"].mean(),
                "t_mean_std_across_images": sub["t_mean"].std(),
                "tau_mean": sub["tau_mean"].mean(),
                "beta_app_m_inv": sub["beta_app_m_inv"].mean(),
                "mi_bits_mean": sub["mi_bits"].mean(),
                "input_psnr_mean": sub["input_psnr"].mean(),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary_path = tables / "split_summary_physics.csv"
    summary.to_csv(summary_path, index=False)

    psd_path = tables / "psd_radial_profiles.csv"
    psd_df.to_csv(psd_path, index=False)
    proxy_path = tables / "fog_proxy_statistics_summary.csv"
    proxy_df.to_csv(proxy_path, index=False)
    return {
        "paired": paired_path,
        "test": test_path,
        "category": cat_path,
        "summary": summary_path,
        "psd": psd_path,
        "proxy": proxy_path,
        "correlations": corr_path,
    }


def add_panel_label(ax, text: str) -> None:
    ax.text(
        0.02,
        0.96,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        fontweight="bold",
        color="white",
        bbox=dict(facecolor="black", alpha=0.65, edgecolor="none", pad=3),
    )


def make_figures(paired_df: pd.DataFrame, metrics_df: pd.DataFrame, psd_df: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    figs = out_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    fig_paths: dict[str, Path] = {}

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for split, color in [("train", "#4c78a8"), ("test", "#f58518")]:
        vals = paired_df.loc[paired_df["split"] == split, "t_mean"].dropna()
        ax.hist(vals, bins=40, alpha=0.55, density=True, label=f"{split} (n={len(vals)})", color=color)
    ax.set_xlabel("Mean apparent transmission per image")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.set_title("Paired chamber apparent transmission")
    fig.tight_layout()
    path = figs / "transmission_distribution.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    fig_paths["transmission_distribution"] = path

    mean_t = np.load(out_dir / "mean_apparent_transmission_map.npy")
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(mean_t, cmap="viridis", vmin=np.nanquantile(mean_t, 0.02), vmax=np.nanquantile(mean_t, 0.98))
    ax.set_axis_off()
    ax.set_title("Mean apparent transmission map")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = figs / "mean_transmission_map.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    fig_paths["mean_transmission_map"] = path

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    order = sorted(paired_df["category"].unique())
    data = [paired_df.loc[(paired_df["split"] == "test") & (paired_df["category"] == c), "beta_app_m_inv"].dropna() for c in order]
    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.set_ylabel("Apparent extinction coefficient (m$^{-1}$)")
    ax.set_title("Apparent optical depth by category")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path = figs / "apparent_extinction_by_category.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    fig_paths["apparent_extinction_by_category"] = path

    # Choose a representative low-performing DCP example from the existing DCP table if available.
    dcp_csv = Path("outputs/dark_channel_prior_fog_chamber_benchmark_20260604/dark_channel_prior_all_metrics.csv")
    if dcp_csv.exists():
        dcp_df = pd.read_csv(dcp_csv)
        dcp_df["dcp_delta_psnr"] = dcp_df["dcp_psnr"] - dcp_df["input_psnr"]
        row = dcp_df.sort_values("dcp_delta_psnr").iloc[len(dcp_df) // 8]
        fog = read_rgb(Path(row["input_path"]), size=384)
        clear = read_rgb(Path(row["gt_path"]), size=384)
    else:
        row = paired_df[paired_df["split"] == "test"].iloc[0]
        fog = read_rgb(Path(row["fog_path"]), size=384)
        clear = read_rgb(Path(row["clear_path"]), size=384)
    air = estimate_airlight(fog)
    empirical_t = apparent_transmission(fog, clear, air)
    dcp_t, _ = dark_channel_prior_transmission(fog)
    fig, axs = plt.subplots(1, 4, figsize=(12.2, 3.3))
    panels = [
        (fog, "Foggy input"),
        (clear, "Clear target"),
        (empirical_t, "Apparent T from pair"),
        (dcp_t, "DCP estimated T"),
    ]
    for ax, (img, title) in zip(axs, panels):
        if img.ndim == 2:
            ax.imshow(img, cmap="magma", vmin=np.nanquantile(img, 0.02), vmax=np.nanquantile(img, 0.98))
        else:
            ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
    fig.tight_layout()
    path = figs / "dcp_vs_empirical_transmission_example.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    fig_paths["dcp_vs_empirical_transmission_example"] = path

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    f = psd_df["frequency_cycles_per_pixel"]
    ax.plot(f, psd_df["chamber_fog_to_clear_ratio"], label="Chamber fog / paired clear", lw=2.0)
    ax.plot(
        f,
        psd_df["outdoor_fog_to_clear_dashcam_proxy_ratio"],
        label="Outdoor fog / clear dashcam proxy",
        lw=2.0,
        ls="--",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Spatial frequency (cycles/pixel)")
    ax.set_ylabel("Power ratio")
    ax.set_title("Frequency-domain fog effect")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = figs / "psd_ratio_chamber_and_outdoor_proxy.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    fig_paths["psd_ratio_chamber_and_outdoor_proxy"] = path

    test = paired_df[paired_df["split"] == "test"].merge(
        metrics_df[["category", "image_name", "nafnet_psnr"]], on=["category", "image_name"], how="left"
    )
    pearson = stats.pearsonr(test["mi_bits"], test["nafnet_psnr"])
    spearman = stats.spearmanr(test["mi_bits"], test["nafnet_psnr"])
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for category, sub in test.groupby("category"):
        ax.scatter(sub["mi_bits"], sub["nafnet_psnr"], s=22, alpha=0.8, label=category)
    ax.set_xlabel("Mutual information, clear vs foggy (bits)")
    ax.set_ylabel("NAFNet PSNR (dB)")
    ax.set_title(f"Information retained vs restoration quality\nPearson r={pearson.statistic:.2f}; Spearman rho={spearman.statistic:.2f}")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    path = figs / "mutual_information_vs_nafnet_psnr.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    fig_paths["mutual_information_vs_nafnet_psnr"] = path

    category = pd.read_csv(out_dir / "tables" / "per_category_physics_and_restoration.csv")
    x = np.arange(len(category))
    fig, ax1 = plt.subplots(figsize=(7.4, 4.2))
    ax1.bar(x - 0.18, category["input_psnr_mean"], width=0.36, label="Foggy input PSNR", color="#bab0ac")
    ax1.bar(x + 0.18, category["nafnet_psnr_mean"], width=0.36, label="NAFNet PSNR", color="#4c78a8")
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(category["category"], rotation=25, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, category["t_mean"], color="#f58518", marker="o", label="Mean apparent T")
    ax2.set_ylabel("Mean apparent transmission")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=8, loc="upper left")
    ax1.set_title("Per-category restoration and fog physics")
    fig.tight_layout()
    path = figs / "per_category_restoration_and_physics.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    fig_paths["per_category_restoration_and_physics"] = path

    corr_path = out_dir / "tables" / "fog_statistic_psnr_correlations.csv"
    if corr_path.exists():
        corr = pd.read_csv(corr_path)
        input_corr = corr[corr["metric_group"] == "input_only"].sort_values("abs_spearman_rho", ascending=False)
        paired_corr = corr[corr["metric_group"] == "paired_diagnostic"].sort_values("abs_spearman_rho", ascending=False)
        plot_corr = pd.concat([input_corr.head(8), paired_corr.head(5)], ignore_index=True)
        colors = ["#4c78a8" if g == "input_only" else "#f58518" for g in plot_corr["metric_group"]]
        fig, ax = plt.subplots(figsize=(8.4, 5.0))
        y_pos = np.arange(len(plot_corr))[::-1]
        ax.barh(y_pos, plot_corr["abs_spearman_rho"], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_corr["metric_label"], fontsize=8)
        ax.set_xlabel("|Spearman rho| with NAFNet PSNR")
        ax.set_title("Fog/input statistic correlation with restoration PSNR")
        ax.set_xlim(0, max(0.05, float(plot_corr["abs_spearman_rho"].max()) * 1.12))
        for y, rho, signed in zip(y_pos, plot_corr["abs_spearman_rho"], plot_corr["spearman_rho"]):
            ax.text(rho + 0.01, y, f"{signed:+.2f}", va="center", fontsize=8)
        ax.text(0.98, 0.03, "blue=input-only; orange=paired diagnostic", transform=ax.transAxes, ha="right", fontsize=8)
        fig.tight_layout()
        path = figs / "fog_statistic_psnr_correlation_rank.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        fig_paths["fog_statistic_psnr_correlation_rank"] = path

        if not input_corr.empty:
            best = input_corr.iloc[0]
            best_col = best["column"]
            best_label = best["metric_label"]
            test_for_scatter = paired_df[paired_df["split"] == "test"].merge(
                metrics_df[["category", "image_name", "nafnet_psnr"]],
                on=["category", "image_name"],
                how="left",
            )
            fig, ax = plt.subplots(figsize=(6.5, 4.8))
            for category_name, sub in test_for_scatter.groupby("category"):
                ax.scatter(sub[best_col], sub["nafnet_psnr"], s=22, alpha=0.78, label=category_name)
            xvals = test_for_scatter[best_col].to_numpy(dtype=np.float64)
            yvals = test_for_scatter["nafnet_psnr"].to_numpy(dtype=np.float64)
            valid = np.isfinite(xvals) & np.isfinite(yvals)
            if int(valid.sum()) >= 3:
                slope, intercept = np.polyfit(xvals[valid], yvals[valid], 1)
                xs = np.linspace(float(np.nanmin(xvals[valid])), float(np.nanmax(xvals[valid])), 100)
                ax.plot(xs, slope * xs + intercept, color="black", lw=1.2, alpha=0.8)
            ax.set_xlabel(best_label)
            ax.set_ylabel("NAFNet PSNR (dB)")
            ax.set_title(f"Best input-only statistic: Spearman rho={best['spearman_rho']:+.2f}")
            ax.legend(frameon=False, fontsize=8, ncol=2)
            fig.tight_layout()
            path = figs / "best_input_fog_statistic_vs_psnr.png"
            fig.savefig(path, dpi=220)
            plt.close(fig)
            fig_paths["best_input_fog_statistic_vs_psnr"] = path

    rayleigh = (1.0 / RGB_WAVELENGTH_NM**4)
    rayleigh = rayleigh / rayleigh.mean() * TUNED_AIRLIGHT.mean()
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(5.7, 3.8))
    ax.bar(x - 0.17, TUNED_AIRLIGHT, width=0.34, label="Tuned simulator airlight", color="#4c78a8")
    ax.bar(x + 0.17, rayleigh, width=0.34, label="Rayleigh shape, mean-matched", color="#e45756")
    ax.set_xticks(x)
    ax.set_xticklabels(["R", "G", "B"])
    ax.set_ylabel("Relative channel value")
    ax.set_title("Airlight color context")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = figs / "airlight_spectrum_context.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    fig_paths["airlight_spectrum_context"] = path

    return fig_paths


def fmt(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.{digits}f}"


def write_report(
    args: argparse.Namespace,
    paired_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    psd_df: pd.DataFrame,
    proxy_df: pd.DataFrame,
    table_paths: dict[str, Path],
    fig_paths: dict[str, Path],
    out_dir: Path,
) -> Path:
    report = out_dir / "fog_physics_report.md"
    summary = pd.read_csv(table_paths["summary"])
    cat = pd.read_csv(table_paths["category"])
    test = paired_df[paired_df["split"] == "test"].merge(
        metrics_df[["category", "image_name", "nafnet_psnr", "nafnet_ssim"]], on=["category", "image_name"], how="left"
    )
    train = summary[summary["split"] == "train"].iloc[0]
    test_summary = summary[summary["split"] == "test"].iloc[0]
    pearson = stats.pearsonr(test["mi_bits"], test["nafnet_psnr"])
    spearman = stats.spearmanr(test["mi_bits"], test["nafnet_psnr"])
    chamber_ratio_low = float(psd_df.loc[psd_df["frequency_cycles_per_pixel"].between(0.01, 0.05), "chamber_fog_to_clear_ratio"].median())
    chamber_ratio_high = float(psd_df.loc[psd_df["frequency_cycles_per_pixel"].between(0.15, 0.24), "chamber_fog_to_clear_ratio"].median())
    outdoor_ratio_high = float(
        psd_df.loc[psd_df["frequency_cycles_per_pixel"].between(0.15, 0.24), "outdoor_fog_to_clear_dashcam_proxy_ratio"].median()
    )
    proxy_idx = proxy_df.set_index("condition")
    chamber_proxy = proxy_idx.loc["Benchmark chamber"]
    outdoor_clear_proxy = proxy_idx.loc["Outdoor clear"]
    outdoor_fog_proxy = proxy_idx.loc["Outdoor fog"]
    corr = pd.read_csv(table_paths["correlations"])
    input_corr = corr[corr["metric_group"] == "input_only"].sort_values("abs_spearman_rho", ascending=False)
    paired_corr = corr[corr["metric_group"] == "paired_diagnostic"].sort_values("abs_spearman_rho", ascending=False)
    best_input_corr = input_corr.iloc[0]
    second_input_corr = input_corr.iloc[1]
    third_input_corr = input_corr.iloc[2]
    best_paired_corr = paired_corr.iloc[0]
    mean_t_map = np.load(out_dir / "mean_apparent_transmission_map.npy")
    h, w = mean_t_map.shape
    yy, xx = np.indices(mean_t_map.shape)
    rr = np.sqrt((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2)
    center_mask = rr < min(h, w) * 0.18
    edge_mask = rr > min(h, w) * 0.43
    map_mean = float(np.nanmean(mean_t_map))
    map_std = float(np.nanstd(mean_t_map))
    map_cv = map_std / max(map_mean, 1e-9)
    map_center = float(np.nanmean(mean_t_map[center_mask]))
    map_edge = float(np.nanmean(mean_t_map[edge_mask]))
    valid_mean = float(test["t_valid_fraction"].mean())
    valid_median = float(test["t_valid_fraction"].median())
    test_nan_t = int(test["t_mean"].isna().sum())
    all_nan_t = int(paired_df["t_mean"].isna().sum())
    best_cat = cat.sort_values("nafnet_psnr_mean", ascending=False).iloc[0]
    worst_cat = cat.sort_values("nafnet_psnr_mean", ascending=True).iloc[0]
    cat_table_lines = [
        "| Category | N | Input PSNR | NAFNet PSNR | NAFNet SSIM | Mean T | Apparent beta | MI bits |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cat.sort_values("category").itertuples(index=False):
        cat_table_lines.append(
            f"| {row.category} | {int(row.n)} | {row.input_psnr_mean:.2f} | "
            f"{row.nafnet_psnr_mean:.2f} | {row.nafnet_ssim_mean:.3f} | "
            f"{row.t_mean:.3f} | {row.beta_app_m_inv:.2f} | {row.mi_bits_mean:.3f} |"
        )
    cat_table_md = "\n".join(cat_table_lines)

    manifest = {
        "created": "2026-06-05",
        "metrics_csv": str(args.metrics_csv),
        "fog_root": str(args.fog_root),
        "clear_root": str(args.clear_root),
        "outdoor_fog_dir": str(args.outdoor_fog_dir),
        "clear_dashcam_dir": str(args.clear_dashcam_dir),
        "analysis_size": args.analysis_size,
        "chamber_depth_m": CHAMBER_DEPTH_M,
        "chamber_transverse_dimensions_mm": list(CHAMBER_TRANSVERSE_MM),
        "paired_counts": paired_df["split"].value_counts().to_dict(),
        "figures": {k: str(v) for k, v in fig_paths.items()},
        "tables": {k: str(v) for k, v in table_paths.items()},
    }
    (out_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    text = f"""# Fog Physics Report

Generated: 2026-06-05

This report analyzes the paired fog-chamber benchmark separately from the manuscript. It answers the proposed physics questions using the available paired chamber images and clearly marks analyses that are only proxies because no aligned clear outdoor targets exist.

## Inputs

- Paired chamber fog images: `{args.fog_root}`
- Paired chamber clear targets: `{args.clear_root}`
- NAFNet held-out metrics and split list: `{args.metrics_csv}`
- Outdoor fog frames without aligned clear targets: `{args.outdoor_fog_dir}`
- Clear dashcam reference images for proxy comparisons: `{args.clear_dashcam_dir}`
- Chamber geometry supplied by measurement: {CHAMBER_DEPTH_M * 1000:.0f} mm fog path length and {CHAMBER_TRANSVERSE_MM[0]} x {CHAMBER_TRANSVERSE_MM[1]} mm transverse dimensions.
- Aggregate image analyses were computed at {args.analysis_size} x {args.analysis_size} pixels to keep the run reproducible without high memory use.

## Executive Summary

- The complete resolved chamber set contains {len(paired_df):,} paired images: {int(train['n']):,} training/non-test images and {int(test_summary['n']):,} held-out benchmark images.
- The held-out benchmark split has mean apparent transmission {fmt(test_summary['t_mean'])}, mean apparent optical depth {fmt(test_summary['tau_mean'])}, and mean apparent extinction coefficient {fmt(test_summary['beta_app_m_inv'])} m^-1 over a {CHAMBER_DEPTH_M:.3f} m path.
- The training/non-test split is similar: mean apparent transmission {fmt(train['t_mean'])}, mean apparent optical depth {fmt(train['tau_mean'])}, and mean apparent extinction coefficient {fmt(train['beta_app_m_inv'])} m^-1.
- The mean apparent-transmission map is not spatially flat: the central region averaged {fmt(map_center)} while the edge region averaged {fmt(map_edge)}, consistent with measurable chamber/camera/display nonuniformity.
- These are apparent sRGB-domain values, not calibrated radiometric extinction coefficients. They are still useful for relative chamber characterization and reproducibility.
- Mutual information between clear and foggy held-out images correlates with NAFNet PSNR with Pearson r={pearson.statistic:.2f} and Spearman rho={spearman.statistic:.2f}, so retained image information is associated with restoration quality but does not by itself define an upper bound.

## Starting Point: Existing Fog Proxy Statistics

The supplement computed fog proxy statistics from the 552-image held-out chamber split and from independent outdoor clear/fog groups. This report reproduces those definitions before adding paired transmission estimates. Images were resized for throughput, the dark-channel haze index was the image mean of a 15-pixel minimum-filtered color-channel dark channel, RMS contrast was luminance standard deviation divided by mean luminance, sharpness was the mean finite-difference luminance gradient magnitude, and airlight shift was the change in mean luminance relative to the available clear reference.

Chamber statistics used paired fog/clear archive images, so contrast, luminance-standard-deviation, and gradient attenuation are fog-over-clear ratios. Outdoor captures are independent groups, so outdoor attenuation is a group-level fog/clear ratio rather than a paired measurement.

Recomputed proxy results: chamber RMS contrast was {fmt(chamber_proxy['rms_contrast_ratio_vs_reference_mean'])} of its paired clear reference and mean gradient was {fmt(chamber_proxy['gradient_ratio_vs_reference_mean'])} of clear. Outdoor fog increased the dark-channel haze index from {fmt(outdoor_clear_proxy['dark_channel_haze_index_mean'])} to {fmt(outdoor_fog_proxy['dark_channel_haze_index_mean'])}, reduced RMS contrast to {fmt(outdoor_fog_proxy['rms_contrast_ratio_vs_reference_mean'])} of the outdoor clear group, and reduced mean gradient to {fmt(outdoor_fog_proxy['gradient_ratio_vs_reference_mean'])} of the outdoor clear group.

Table: `{table_paths['proxy']}`.

## Which Fog Statistic Predicts Restoration PSNR?

Candidate statistics were screened against NAFNet PSNR on the 552-image held-out split. The screen included input-only statistics computed from the foggy image alone and paired diagnostics that require the aligned clear target. Raw Pearson and Spearman correlations were computed, and category-centered correlations were computed by subtracting category means from both the statistic and NAFNet PSNR.

The best input-only statistic was the {best_input_corr['metric_label']}, with Spearman rho={best_input_corr['spearman_rho']:.3f} (p={best_input_corr['spearman_p']:.2e}) and Pearson r={best_input_corr['pearson_r']:.3f} (p={best_input_corr['pearson_p']:.2e}). After category centering, the relationship remained but weakened to Spearman rho={best_input_corr['category_centered_spearman_rho']:.3f} (p={best_input_corr['category_centered_spearman_p']:.2e}). The next-best input-only statistics were {second_input_corr['metric_label']} (rho={second_input_corr['spearman_rho']:.3f}) and {third_input_corr['metric_label']} (rho={third_input_corr['spearman_rho']:.3f}). These correlations are useful but moderate, so the top input-only statistic should be described as the best tested input-only correlate of restoration PSNR, not as a calibrated fog-density measurement.

The strongest paired diagnostic was {best_paired_corr['metric_label']}, with Spearman rho={best_paired_corr['spearman_rho']:.3f} (p={best_paired_corr['spearman_p']:.2e}) and category-centered Spearman rho={best_paired_corr['category_centered_spearman_rho']:.3f} (p={best_paired_corr['category_centered_spearman_p']:.2e}). This is more physically interpretable for the benchmark: images that retain more of the clear target's gradient energy after fogging are easier for NAFNet to restore. Because this statistic requires the aligned clear target, it is useful for benchmark analysis and dataset characterization but cannot be used as an input-only fog-density estimator.

Correlation table: `{table_paths['correlations']}`.

Figures:

- `{fig_paths['fog_statistic_psnr_correlation_rank']}`
- `{fig_paths['best_input_fog_statistic_vs_psnr']}`

## A. Direct Transmission Map Estimation

For each paired chamber image, airlight was estimated from the foggy image using the dark-channel candidate rule: the top 0.1% of dark-channel pixels were selected and the brightest candidate was used as airlight. Apparent transmission was then estimated from the paired equation

`I(x) = J(x) T(x) + A(1 - T(x))`

using the median of valid per-channel estimates of `(I - A) / (J - A)`. Pixels with unstable denominators or nonphysical outliers were excluded. This produces an empirical, apparent transmission map without assuming scene depth.

Key result: the held-out split has mean apparent transmission {fmt(test_summary['t_mean'])} +/- {fmt(test_summary['t_mean_std_across_images'])} across images. The mean spatial map is saved as `{fig_paths['mean_transmission_map']}` and the train/test distribution is saved as `{fig_paths['transmission_distribution']}`.

Quality control: the mean valid pixel fraction for the held-out split was {fmt(valid_mean)} and the median was {fmt(valid_median)} after excluding unstable or nonphysical pixelwise estimates. No held-out images lacked a valid mean transmission estimate; {all_nan_t} image across the full 5,495-image resolved set lacked a valid mean after masking. The 512 x 512 mean apparent-transmission map had spatial coefficient of variation {fmt(map_cv)}. Using a central disk and an outer-edge region as simple spatial summaries, the center averaged {fmt(map_center)} and the edge averaged {fmt(map_edge)}.

Interpretation: the chamber fog is sufficiently uniform to support a single controlled benchmark condition, but the empirical maps are not perfectly flat. The residual spatial structure is expected because the measurement combines fog, camera response, display emission, alignment, and chamber optics.

## B. Apparent Optical Depth and Extinction Coefficient

Using the measured chamber depth of {CHAMBER_DEPTH_M:.3f} m, apparent optical depth was computed as `tau = -log(T)` and apparent extinction as `beta_app = tau / {CHAMBER_DEPTH_M:.3f}`. On the held-out split, mean apparent `tau` is {fmt(test_summary['tau_mean'])} and mean apparent `beta_app` is {fmt(test_summary['beta_app_m_inv'])} m^-1.

This is a useful benchmark descriptor, but it should not be presented as a calibrated physical extinction coefficient unless the display, camera response, exposure, and spectral radiance are calibrated. In the manuscript, the safe wording is "apparent optical depth" or "sRGB-domain apparent extinction."

## C. Airlight Spectrum

The tuned simulator airlight was R={TUNED_AIRLIGHT[0]:.3f}, G={TUNED_AIRLIGHT[1]:.3f}, B={TUNED_AIRLIGHT[2]:.3f}. The blue/red ratio is {TUNED_AIRLIGHT[2] / TUNED_AIRLIGHT[0]:.2f}. A Rayleigh-like wavelength^-4 curve over nominal R/G/B wavelengths would be much more blue weighted than this near-neutral airlight, as shown in `{fig_paths['airlight_spectrum_context']}`.

Conclusion: the fitted airlight is consistent with a nearly neutral, large-particle/artificial-fog scattering appearance and is not Rayleigh dominated. It is not enough to infer a 2-5 um particle size without spectral calibration or a measured particle-size distribution.

## D. Frequency-Domain Fog Characterization

For the 552 held-out paired chamber images, the mean radial power spectrum of foggy images was divided by the mean radial power spectrum of the paired clear targets. The median chamber fog/clear power ratio was {fmt(chamber_ratio_low)} at low spatial frequencies and {fmt(chamber_ratio_high)} at higher spatial frequencies. This supports the expected interpretation that chamber fog suppresses higher spatial frequencies more strongly than lower frequencies.

For the outdoor fog frames, no aligned clear references are available. The report therefore compares outdoor fog PSD to the separate clear dashcam reference set as a proxy only. The high-frequency outdoor proxy ratio is {fmt(outdoor_ratio_high)}. This is useful for checking whether outdoor examples have similar frequency content, but it is not a paired degradation measurement.

Architecture implication: the PSD result supports including multi-scale or frequency-aware restoration models in the benchmark, but it does not by itself prove that any one architecture class is physically preferred. It is better used as explanatory context for why fine detail is difficult to recover than as a model-selection claim.

Figure: `{fig_paths['psd_ratio_chamber_and_outdoor_proxy']}`.

## E. Mutual Information vs. Restoration Quality

Mutual information was estimated from binned clear/foggy luminance histograms on each held-out pair. NAFNet PSNR increases with mutual information: Pearson r={pearson.statistic:.2f} (p={pearson.pvalue:.2e}) and Spearman rho={spearman.statistic:.2f} (p={spearman.pvalue:.2e}).

This supports the idea that retained information in the foggy image is one constraint on achievable restoration quality, but the relationship is weak. The current data therefore do not support the stronger claim that information content is the binding constraint on NAFNet performance. Content category, texture, camera/display effects, and model behavior likely account for substantial remaining variance. The estimate also does not establish a true information-theoretic upper bound because it is binned, sRGB-domain, and does not model spatial dependencies or camera noise.

Figure: `{fig_paths['mutual_information_vs_nafnet_psnr']}`.

## F. Per-Category Restoration Analysis

The held-out set has six categories with 92 images each. The easiest category by NAFNet PSNR is `{best_cat['category']}` at {fmt(best_cat['nafnet_psnr_mean'])} dB; the hardest is `{worst_cat['category']}` at {fmt(worst_cat['nafnet_psnr_mean'])} dB. Category-level restoration does not simply follow mean apparent transmission, so image content and texture also matter.

{cat_table_md}

Table: `{table_paths['category']}`.

Figure: `{fig_paths['per_category_restoration_and_physics']}`.

## DCP Failure Interpretation

The dark channel prior estimates a transmission map from local low-channel statistics in the foggy image. In the chamber, the paired apparent-transmission diagnostic is affected by display content, camera response, chamber nonuniformity, and optics rather than by outdoor scene depth alone. The example in `{fig_paths['dcp_vs_empirical_transmission_example']}` shows that the DCP transmission estimate can differ strongly from the paired sRGB-domain diagnostic. This supports the interpretation that DCP's outdoor-scene assumptions are mismatched to the chamber benchmark, without treating the paired map as calibrated physical ground truth.

## Report Tables

- Per-image apparent transmission and MI: `{table_paths['paired']}`
- Held-out physics merged with NAFNet metrics: `{table_paths['test']}`
- Per-category physics and restoration: `{table_paths['category']}`
- Train/test physics summary: `{table_paths['summary']}`
- Radial PSD profiles: `{table_paths['psd']}`
- Recomputed fog proxy statistics: `{table_paths['proxy']}`
- Fog/input statistic correlations with NAFNet PSNR: `{table_paths['correlations']}`

## Recommended Manuscript Use

Use only conservative claims unless a full calibration is added:

- Good: "The chamber has a measured 114 mm fog path and apparent optical-depth statistics computed from paired clear/fog images."
- Good: "Paired sRGB analysis shows nonuniform but reproducible apparent transmission across the benchmark."
- Good: "Frequency-domain and mutual-information analyses support the interpretation that fog suppresses high-frequency detail and that retained information correlates with restoration quality."
- Avoid: "The chamber is physically calibrated" or "the extinction coefficient is a true atmospheric beta" unless radiometric calibration is performed.
- Avoid: "Outdoor transmission was directly estimated" because the outdoor fog images are unpaired.

## Best Parts to Add to the Defogging Paper

The strongest additions for the paper are the ones that strengthen the dataset and evaluation story without overclaiming physical calibration.

1. Add the PSNR-correlation screen as the main fog-characterization result. The most useful concise claim is: "Across {len(input_corr)} tested input-only fog statistics, {best_input_corr['metric_label'].lower()} had the strongest association with NAFNet PSNR on the 552-image held-out split (Spearman rho={best_input_corr['spearman_rho']:.3f}; category-centered rho={best_input_corr['category_centered_spearman_rho']:.3f}). The strongest paired diagnostic was fog/clear gradient retention (rho={best_paired_corr['spearman_rho']:.3f}), indicating that restoration quality is highest when fogging preserves clear-image gradient structure." This is more useful than reporting a disconnected list of fog statistics.

2. Add the chamber dimensions and apparent optical-depth summary to the methods or supplement. The most useful concise claim is: "The chamber had a 114 mm fog path length and 133 x 114 mm transverse dimensions. Paired sRGB analysis of the 552-image held-out split gave mean apparent transmission 0.486 and mean apparent optical depth 0.854." Keep the word "apparent" because the camera/display system is not radiometrically calibrated.

3. Add the mean apparent-transmission map as a supplement figure or as a small panel in an existing supplement fog-statistics figure. This is the most visually useful physics result. It shows that the chamber condition is reproducible but not spatially flat: the central region averaged 0.405 while the edge region averaged 0.582. This is better than only saying "fog density proxy" because it uses the paired structure of the dataset.

4. Add the DCP-vs-paired-transmission diagnostic to the DCP supplement section if space allows. It gives a concrete explanation for why DCP fails: DCP estimates an outdoor-style transmission structure from local dark-channel statistics, while the chamber diagnostic reflects display content, chamber nonuniformity, optics, and camera response. This should be framed as a diagnostic, not as calibrated ground-truth transmission.

5. Add the per-category restoration table to the supplement. It is low risk and useful: all categories have 92 held-out examples, furniture is easiest for NAFNet at 26.44 dB / 0.846 SSIM, and artwork is hardest at 23.02 dB / 0.756 SSIM. This gives reviewers a better sense of what content types drive the benchmark.

6. Mention the PSD result briefly as explanatory context, not as a new headline result. The held-out chamber fog/clear power ratio fell from 0.054 at low spatial frequencies to 0.004 at higher spatial frequencies, supporting the expected conclusion that fog strongly suppresses fine detail. This can motivate why restoration is difficult and why multi-scale/frequency-aware models are reasonable benchmark candidates.

7. Keep the MI result in the supplement only, or omit it from the paper unless space is available. It is analytically interesting, but the correlation with NAFNet PSNR is weak (Pearson r=0.18; Spearman rho=0.16). The honest conclusion is that retained information contributes to restoration quality but is not the sole binding constraint.

I would not add a particle-size or Mie-scattering claim to the paper from this report alone. The airlight color is useful as simulator provenance, but it does not justify a particle-size estimate without spectral calibration or independent particle measurements.
"""
    report.write_text(text)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("results/nafnet_runs/metrics.csv"),
    )
    parser.add_argument(
        "--fog-root",
        type=Path,
        default=Path("VerticalFilter_MediumFog_Redo_3-21-26_aligned"),
    )
    parser.add_argument(
        "--clear-root",
        type=Path,
        default=Path("archive_gt_matched"),
    )
    parser.add_argument(
        "--outdoor-fog-dir",
        type=Path,
        default=Path("fog_all_data/3_29_26/LightFogSenterra_OutdoorCapture_3-29-26"),
    )
    parser.add_argument(
        "--clear-dashcam-dir",
        type=Path,
        default=Path("fog_all_data/3_29_26/Senterra_OutdoorCapture_3-29-26"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./outputs_20260605"),
    )
    parser.add_argument("--analysis-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "tables").mkdir(exist_ok=True)
    (args.out_dir / "figures").mkdir(exist_ok=True)
    metrics_df = load_metrics(args.metrics_csv)
    records = resolve_pairs(args.metrics_csv, args.fog_root, args.clear_root)
    if not records:
        raise RuntimeError("No paired chamber records resolved")
    print(f"resolved {len(records)} paired chamber images")
    paired_df = analyze_pairs(records, args.analysis_size, args.out_dir)
    psd_df = analyze_psd(paired_df, args.outdoor_fog_dir, args.clear_dashcam_dir, args.analysis_size)
    proxy_df = summarize_proxy_metrics(paired_df, args.outdoor_fog_dir, args.clear_dashcam_dir, args.analysis_size)
    table_paths = write_csvs(paired_df, metrics_df, psd_df, proxy_df, args.out_dir)
    fig_paths = make_figures(paired_df, metrics_df, psd_df, args.out_dir)
    report_path = write_report(args, paired_df, metrics_df, psd_df, proxy_df, table_paths, fig_paths, args.out_dir)
    print(f"report: {report_path}")
    print(json.dumps({"tables": {k: str(v) for k, v in table_paths.items()}, "figures": {k: str(v) for k, v in fig_paths.items()}}, indent=2))


if __name__ == "__main__":
    main()
