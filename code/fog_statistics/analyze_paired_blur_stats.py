#!/usr/bin/env python3
"""Paired blur / PSF-equivalent statistics for fog-chamber images.

These are diagnostic metrics, not a calibrated optical PSF. The main estimate
fits an equivalent Gaussian blur to the radial power-spectrum ratio between
foggy and clear images:

    P_fog(f) / P_clear(f) ~= exp(-4*pi^2*sigma^2*f^2)

where f is cycles/pixel. The reported FWHM is 2.355*sigma in pixels.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy import ndimage, stats


ROOT = Path(".")
BASE_TABLE = ROOT / "outputs_20260605" / "tables" / "test_physics_plus_nafnet_metrics.csv"
OUTDIR = ROOT / "outputs_20260605" / "paired_blur_stats"
SIZE = 512
DIFF_STEPS = (1, 2, 3, 4, 6, 8, 12)
GAUSSIAN_SIGMAS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def read_rgb(path: str | Path) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if img.size != (SIZE, SIZE):
        img = img.resize((SIZE, SIZE), Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


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
    edges = np.linspace(0.0, rmax, bins + 1)
    which = np.digitize(radius.reshape(-1), edges) - 1
    sums = np.bincount(which, weights=power.reshape(-1), minlength=bins)
    counts = np.bincount(which, minlength=bins)
    radial = sums[:bins] / np.maximum(counts[:bins], 1)
    freq = 0.5 * (edges[:-1] + edges[1:]) / min(h, w)
    return freq.astype(np.float64), radial.astype(np.float64)


def gradient(gray: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(gray)
    return np.sqrt(gx * gx + gy * gy)


def stepped_gradient(gray: np.ndarray, step: int) -> np.ndarray:
    gx = gray[:, step:] - gray[:, :-step]
    gy = gray[step:, :] - gray[:-step, :]
    return np.concatenate((gx.reshape(-1), gy.reshape(-1))) / float(step)


def stepped_laplacian(gray: np.ndarray, step: int) -> np.ndarray:
    center = gray[step:-step, step:-step]
    lap = (
        gray[step:-step, 2 * step :]
        + gray[step:-step, : -2 * step]
        + gray[2 * step :, step:-step]
        + gray[: -2 * step, step:-step]
        - 4.0 * center
    )
    return lap / float(step * step)


def gaussian_gradient(gray: np.ndarray, sigma: float) -> np.ndarray:
    gx = ndimage.gaussian_filter(gray, sigma=sigma, order=(0, 1), mode="reflect")
    gy = ndimage.gaussian_filter(gray, sigma=sigma, order=(1, 0), mode="reflect")
    return np.sqrt(gx * gx + gy * gy)


def ratio(numer: float, denom: float, eps: float = 1e-12) -> float:
    return float(numer / max(denom, eps))


def multiscale_derivative_metrics(clear: np.ndarray, fog: np.ndarray) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for step in DIFF_STEPS:
        clear_grad = stepped_gradient(clear, step)
        fog_grad = stepped_gradient(fog, step)
        clear_lap = stepped_laplacian(clear, step)
        fog_lap = stepped_laplacian(fog, step)
        metrics[f"gradient_step{step}_mean_abs_ratio"] = ratio(
            float(np.mean(np.abs(fog_grad))),
            float(np.mean(np.abs(clear_grad))),
        )
        metrics[f"gradient_step{step}_energy_ratio"] = ratio(
            float(np.mean(fog_grad * fog_grad)),
            float(np.mean(clear_grad * clear_grad)),
        )
        metrics[f"laplacian_step{step}_variance_ratio"] = ratio(
            float(np.var(fog_lap)),
            float(np.var(clear_lap)),
        )
        metrics[f"laplacian_step{step}_mean_abs_ratio"] = ratio(
            float(np.mean(np.abs(fog_lap))),
            float(np.mean(np.abs(clear_lap))),
        )

    for sigma in GAUSSIAN_SIGMAS:
        key = str(sigma).replace(".", "p")
        clear_ggrad = gaussian_gradient(clear, sigma)
        fog_ggrad = gaussian_gradient(fog, sigma)
        clear_log = ndimage.gaussian_laplace(clear, sigma=sigma, mode="reflect")
        fog_log = ndimage.gaussian_laplace(fog, sigma=sigma, mode="reflect")
        metrics[f"gaussian_gradient_sigma{key}_mean_ratio"] = ratio(
            float(np.mean(fog_ggrad)),
            float(np.mean(clear_ggrad)),
        )
        metrics[f"gaussian_gradient_sigma{key}_energy_ratio"] = ratio(
            float(np.mean(fog_ggrad * fog_ggrad)),
            float(np.mean(clear_ggrad * clear_ggrad)),
        )
        metrics[f"log_sigma{key}_variance_ratio"] = ratio(
            float(np.var(fog_log)),
            float(np.var(clear_log)),
        )
        metrics[f"log_sigma{key}_mean_abs_ratio"] = ratio(
            float(np.mean(np.abs(fog_log))),
            float(np.mean(np.abs(clear_log))),
        )

    return metrics


def equivalent_gaussian_fwhm(freq: np.ndarray, clear_power: np.ndarray, fog_power: np.ndarray) -> dict[str, float]:
    eps = 1e-18
    ratio = fog_power / np.maximum(clear_power, eps)
    valid = (
        np.isfinite(ratio)
        & (ratio > 1e-5)
        & (ratio < 1.5)
        & (freq >= 0.035)
        & (freq <= 0.22)
        & (clear_power > np.percentile(clear_power, 20))
    )
    if int(valid.sum()) < 8:
        return {
            "equiv_gaussian_sigma_px": np.nan,
            "equiv_gaussian_fwhm_px": np.nan,
            "psd_fit_r2": np.nan,
            "psd_fit_n": float(valid.sum()),
        }
    x = freq[valid] ** 2
    y = np.log(np.clip(ratio[valid], 1e-5, 1.5))
    slope, intercept, r_value, _p_value, _stderr = stats.linregress(x, y)
    # Power spectrum of a Gaussian-blurred image decays as exp(-4*pi^2*sigma^2*f^2).
    sigma = math.sqrt(max(0.0, -slope) / (4.0 * math.pi * math.pi))
    return {
        "equiv_gaussian_sigma_px": float(sigma),
        "equiv_gaussian_fwhm_px": float(2.354820045 * sigma),
        "psd_fit_r2": float(r_value * r_value),
        "psd_fit_n": float(valid.sum()),
        "psd_fit_slope": float(slope),
        "psd_fit_intercept": float(intercept),
    }


def mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    hist, _, _ = np.histogram2d(a.reshape(-1), b.reshape(-1), bins=bins, range=[[0, 1], [0, 1]])
    pxy = hist / max(hist.sum(), 1.0)
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0
    return float(np.sum(pxy[nz] * np.log2(pxy[nz] / (px @ py)[nz])))


def per_pair(row: pd.Series) -> dict[str, float | str]:
    clear = luminance(read_rgb(row["clear_path"]))
    fog = luminance(read_rgb(row["fog_path"]))

    # Compare structure after removing global brightness/contrast differences.
    clear_z = (clear - clear.mean()) / max(clear.std(), 1e-6)
    fog_z = (fog - fog.mean()) / max(fog.std(), 1e-6)
    residual = fog_z - clear_z

    f, p_clear = radial_power(clear)
    _, p_fog = radial_power(fog)
    _, p_clear_z = radial_power(clear_z)
    _, p_fog_z = radial_power(fog_z)
    high = f >= 0.15
    mid = (f >= 0.05) & (f < 0.15)
    low = f < 0.05

    clear_grad = gradient(clear)
    fog_grad = gradient(fog)
    clear_lap = ndimage.laplace(clear)
    fog_lap = ndimage.laplace(fog)
    fit = equivalent_gaussian_fwhm(f, p_clear_z, p_fog_z)
    scale_metrics = multiscale_derivative_metrics(clear, fog)

    return {
        "category": row["category"],
        "image_name": row["image_name"],
        "nafnet_psnr": float(row["nafnet_psnr"]),
        "input_psnr": float(row["input_psnr"]),
        "gradient_ratio_recomputed": float(fog_grad.mean() / max(clear_grad.mean(), 1e-12)),
        "tenengrad_ratio": float(np.mean(fog_grad * fog_grad) / max(np.mean(clear_grad * clear_grad), 1e-12)),
        "laplacian_variance_ratio": float(np.var(fog_lap) / max(np.var(clear_lap), 1e-12)),
        "brenner_ratio": float(
            (
                np.mean((fog[:, 2:] - fog[:, :-2]) ** 2)
                + np.mean((fog[2:, :] - fog[:-2, :]) ** 2)
            )
            / max(
                (
                    np.mean((clear[:, 2:] - clear[:, :-2]) ** 2)
                    + np.mean((clear[2:, :] - clear[:-2, :]) ** 2)
                ),
                1e-12,
            )
        ),
        "psd_high_power_ratio_raw": float(np.sum(p_fog[high]) / max(np.sum(p_clear[high]), 1e-18)),
        "psd_mid_power_ratio_raw": float(np.sum(p_fog[mid]) / max(np.sum(p_clear[mid]), 1e-18)),
        "psd_low_power_ratio_raw": float(np.sum(p_fog[low]) / max(np.sum(p_clear[low]), 1e-18)),
        "psd_high_power_ratio_zscore": float(np.sum(p_fog_z[high]) / max(np.sum(p_clear_z[high]), 1e-18)),
        "psd_mid_power_ratio_zscore": float(np.sum(p_fog_z[mid]) / max(np.sum(p_clear_z[mid]), 1e-18)),
        "psd_high_mid_ratio_zscore": float(
            (np.sum(p_fog_z[high]) / max(np.sum(p_fog_z[mid]), 1e-18))
            / max((np.sum(p_clear_z[high]) / max(np.sum(p_clear_z[mid]), 1e-18)), 1e-18)
        ),
        "zscore_structural_rmse": float(np.sqrt(np.mean(residual * residual))),
        "zscore_structural_corr": float(np.corrcoef(clear_z.reshape(-1), fog_z.reshape(-1))[0, 1]),
        "clear_fog_mi_bits_recomputed": mutual_information(clear, fog),
        **fit,
        **scale_metrics,
    }


def correlation_rows(df: pd.DataFrame, columns: list[str]) -> list[dict[str, float | str]]:
    rows = []
    y = df["nafnet_psnr"].to_numpy(dtype=float)
    category_centered_y = df["nafnet_psnr"] - df.groupby("category")["nafnet_psnr"].transform("mean")
    for col in columns:
        x = df[col].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 5:
            continue
        cc_x = df[col] - df.groupby("category")[col].transform("mean")
        cc_finite = np.isfinite(cc_x) & np.isfinite(category_centered_y)
        pearson = stats.pearsonr(x[finite], y[finite])
        spearman = stats.spearmanr(x[finite], y[finite])
        cc_pearson = stats.pearsonr(cc_x[cc_finite], category_centered_y[cc_finite])
        cc_spearman = stats.spearmanr(cc_x[cc_finite], category_centered_y[cc_finite])
        rows.append(
            {
                "column": col,
                "n": int(finite.sum()),
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
                "category_centered_pearson_r": float(cc_pearson.statistic),
                "category_centered_pearson_p": float(cc_pearson.pvalue),
                "category_centered_spearman_rho": float(cc_spearman.statistic),
                "category_centered_spearman_p": float(cc_spearman.pvalue),
                "abs_spearman_rho": abs(float(spearman.statistic)),
                "abs_category_centered_spearman_rho": abs(float(cc_spearman.statistic)),
            }
        )
    return sorted(rows, key=lambda r: r["abs_spearman_rho"], reverse=True)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(BASE_TABLE)
    rows = [per_pair(row) for _, row in base.iterrows()]
    paired = pd.DataFrame(rows)
    paired.to_csv(OUTDIR / "paired_blur_psf_equivalent_metrics.csv", index=False)

    metric_cols = [c for c in paired.columns if c not in {"category", "image_name", "nafnet_psnr"}]
    corr = pd.DataFrame(correlation_rows(paired, metric_cols))
    corr.to_csv(OUTDIR / "paired_blur_psf_equivalent_correlations.csv", index=False)

    top = corr.head(12).copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    colors = ["#4267ac" if "fwhm" not in c and "sigma" not in c else "#b64545" for c in top["column"]]
    ax.barh(top["column"][::-1], top["spearman_rho"][::-1], color=colors[::-1])
    ax.axvline(0, color="0.2", linewidth=0.8)
    ax.set_xlabel("Spearman rho with NAFNet PSNR")
    ax.set_title("Paired fog statistics ranked by PSNR correlation")
    fig.tight_layout()
    fig.savefig(OUTDIR / "paired_blur_psf_correlation_rank.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.scatter(paired["equiv_gaussian_fwhm_px"], paired["nafnet_psnr"], s=12, alpha=0.65)
    ax.set_xlabel("Equivalent Gaussian FWHM (pixels)")
    ax.set_ylabel("NAFNet PSNR (dB)")
    ax.set_title("PSF-equivalent blur vs restoration PSNR")
    fig.tight_layout()
    fig.savefig(OUTDIR / "equivalent_fwhm_vs_nafnet_psnr.png", dpi=220)
    plt.close(fig)

    with (OUTDIR / "paired_blur_summary.txt").open("w") as f:
        f.write(f"n={len(paired)}\n")
        f.write("Top paired metrics by absolute Spearman correlation with NAFNet PSNR:\n")
        for _, row in corr.head(12).iterrows():
            f.write(
                f"{row['column']}: rho={row['spearman_rho']:.4f}, "
                f"category-centered rho={row['category_centered_spearman_rho']:.4f}\n"
            )


if __name__ == "__main__":
    main()
