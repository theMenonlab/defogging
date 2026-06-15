#!/usr/bin/env python3
"""Paired image-domain PSD analysis for chamber fog captures.

This analysis compares aligned foggy chamber captures with the archive images
displayed on the screen. It is an end-to-end image degradation measurement, not
a radiometric estimate of physical fog transmission.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
DEFAULT_FOG_ROOT = Path("VerticalFilter_MediumFog_Redo_3-21-26_aligned")
DEFAULT_ARCHIVE_ROOT = Path("archive_gt_matched")
DEFAULT_OUT_DIR = Path("./PSD_calculation")


@dataclass(frozen=True)
class PairRecord:
    split: str
    category: str
    stem: str
    fog_path: Path
    archive_path: Path


def read_rgb(path: Path, size: int) -> np.ndarray:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def radial_power(gray: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = gray.shape
    if h != w:
        raise ValueError(f"Expected square image, got {gray.shape}")
    centered = gray - float(np.mean(gray))
    window = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    spectrum = np.fft.fftshift(np.fft.fft2(centered * window))
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


def resolve_pairs(fog_root: Path, archive_root: Path) -> list[PairRecord]:
    records: list[PairRecord] = []
    categories = sorted(p.name for p in fog_root.iterdir() if p.is_dir())
    for category in categories:
        fog_dir = fog_root / category
        archive_dir = archive_root / category
        if not archive_dir.is_dir():
            raise FileNotFoundError(f"Missing archive category directory: {archive_dir}")

        archive_by_stem = {
            p.stem: p
            for p in archive_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        }
        fog_paths = sorted(
            [p for p in fog_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
            key=lambda p: p.stem,
        )
        for idx, fog_path in enumerate(fog_paths):
            archive_path = archive_by_stem.get(fog_path.stem)
            if archive_path is None:
                raise FileNotFoundError(f"No archive match for {fog_path}")
            split = "test" if idx % 10 == 0 else "train"
            records.append(PairRecord(split, category, fog_path.stem, fog_path, archive_path))
    return records


def band_median(freq: np.ndarray, values: np.ndarray, lo: float, hi: float) -> float:
    mask = (freq >= lo) & (freq <= hi)
    if not np.any(mask):
        raise ValueError(f"No frequency bins in band {lo}-{hi}")
    return float(np.nanmedian(values[..., mask], axis=-1))


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, reps: int = 5000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, values.size, size=(reps, values.size))
    boot = np.median(values[idx], axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def analyze(records: list[PairRecord], size: int, bins: int, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    ratio_rows = []
    freq = None

    selected = [r for r in records if r.split == "test"]
    if not selected:
        raise RuntimeError("No test records after deterministic split")

    fog_power_acc = None
    archive_power_acc = None
    per_pair_ratios = []

    for i, rec in enumerate(selected, start=1):
        fog_lum = luminance(read_rgb(rec.fog_path, size))
        archive_lum = luminance(read_rgb(rec.archive_path, size))
        freq_i, fog_power = radial_power(fog_lum, bins)
        _, archive_power = radial_power(archive_lum, bins)
        if freq is None:
            freq = freq_i
            fog_power_acc = np.zeros_like(fog_power)
            archive_power_acc = np.zeros_like(archive_power)

        eps = 1e-12
        ratio = fog_power / np.maximum(archive_power, eps)
        per_pair_ratios.append(ratio)
        fog_power_acc += fog_power
        archive_power_acc += archive_power
        rows.append(
            {
                "split": rec.split,
                "category": rec.category,
                "stem": rec.stem,
                "fog_path": str(rec.fog_path),
                "archive_path": str(rec.archive_path),
                "low_band_ratio_0p01_0p05": band_median(freq_i, ratio, 0.01, 0.05),
                "high_band_ratio_0p15_0p24": band_median(freq_i, ratio, 0.15, 0.24),
            }
        )
        if i % 100 == 0:
            print(f"processed {i}/{len(selected)} held-out pairs", flush=True)

    assert freq is not None
    assert fog_power_acc is not None
    assert archive_power_acc is not None
    ratio_stack = np.vstack(per_pair_ratios)
    mean_fog_power = fog_power_acc / len(selected)
    mean_archive_power = archive_power_acc / len(selected)

    for bin_idx, f in enumerate(freq):
        bin_ratios = ratio_stack[:, bin_idx]
        ratio_rows.append(
            {
                "frequency_cycles_per_pixel": float(f),
                "median_pair_fog_to_archive_ratio": float(np.median(bin_ratios)),
                "q25_pair_fog_to_archive_ratio": float(np.quantile(bin_ratios, 0.25)),
                "q75_pair_fog_to_archive_ratio": float(np.quantile(bin_ratios, 0.75)),
                "mean_fog_power": float(mean_fog_power[bin_idx]),
                "mean_archive_power": float(mean_archive_power[bin_idx]),
                "mean_power_fog_to_archive_ratio": float(
                    mean_fog_power[bin_idx] / max(mean_archive_power[bin_idx], 1e-12)
                ),
            }
        )

    per_pair_df = pd.DataFrame(rows)
    profile_df = pd.DataFrame(ratio_rows)
    rng = np.random.default_rng(20260611)

    low_pair = per_pair_df["low_band_ratio_0p01_0p05"].to_numpy(dtype=np.float64)
    high_pair = per_pair_df["high_band_ratio_0p15_0p24"].to_numpy(dtype=np.float64)
    low_median = float(np.median(low_pair))
    high_median = float(np.median(high_pair))
    relative_suppression = float(low_median / high_median) if high_median > 0 else float("inf")
    category_summary = (
        per_pair_df.groupby("category", as_index=False)
        .agg(
            n=("stem", "count"),
            low_band_ratio_median=("low_band_ratio_0p01_0p05", "median"),
            high_band_ratio_median=("high_band_ratio_0p15_0p24", "median"),
        )
        .assign(
            low_over_high_relative_suppression=lambda d: d["low_band_ratio_median"]
            / d["high_band_ratio_median"]
        )
    )

    split_counts = pd.DataFrame(
        [{"category": c, "split": s, "n": sum(1 for r in records if r.category == c and r.split == s)}
         for c in sorted({r.category for r in records}) for s in ("train", "test")]
    )
    summary = {
        "analysis": "paired image-domain radial PSD ratio",
        "claim_boundary": (
            "Ratios compare foggy chamber captures with displayed archive images. "
            "They measure end-to-end image degradation, not physical fog transmission."
        ),
        "fog_root": str(DEFAULT_FOG_ROOT),
        "archive_root": str(DEFAULT_ARCHIVE_ROOT),
        "image_size": size,
        "radial_bins": bins,
        "all_pairs": len(records),
        "held_out_pairs": len(selected),
        "split_rule": "within each category, sorted by stem; zero-based index modulo 10 equals 0 is held out",
        "low_band_cycles_per_pixel": [0.01, 0.05],
        "high_band_cycles_per_pixel": [0.15, 0.24],
        "low_band_median_pair_ratio": low_median,
        "low_band_median_pair_ratio_ci95": list(bootstrap_ci(low_pair, rng)),
        "high_band_median_pair_ratio": high_median,
        "high_band_median_pair_ratio_ci95": list(bootstrap_ci(high_pair, rng)),
        "low_over_high_relative_suppression": relative_suppression,
        "category_summary": category_summary.to_dict(orient="records"),
    }

    per_pair_df.to_csv(out_dir / "per_pair_psd_band_ratios.csv", index=False)
    profile_df.to_csv(out_dir / "radial_psd_profile.csv", index=False)
    category_summary.to_csv(out_dir / "category_psd_summary.csv", index=False)
    split_counts.to_csv(out_dir / "split_counts.csv", index=False)
    (out_dir / "psd_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return per_pair_df, profile_df, summary


def make_figure(profile_df: pd.DataFrame, summary: dict, out_dir: Path) -> None:
    freq = profile_df["frequency_cycles_per_pixel"].to_numpy()
    median_ratio = profile_df["median_pair_fog_to_archive_ratio"].to_numpy()
    q25 = profile_df["q25_pair_fog_to_archive_ratio"].to_numpy()
    q75 = profile_df["q75_pair_fog_to_archive_ratio"].to_numpy()
    mean_ratio = profile_df["mean_power_fog_to_archive_ratio"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.fill_between(freq, q25, q75, color="#b9d6f2", alpha=0.55, linewidth=0, label="Pair IQR")
    ax.plot(freq, median_ratio, color="#1f5a92", lw=2.2, label="Median pair ratio")
    ax.plot(freq, mean_ratio, color="#b04a2f", lw=1.6, ls="--", label="Mean PSD ratio")
    ax.axvspan(0.01, 0.05, color="#5c946e", alpha=0.13, linewidth=0)
    ax.axvspan(0.15, 0.24, color="#c36f09", alpha=0.13, linewidth=0)
    ax.set_yscale("log")
    ax.set_xlim(0, 0.25)
    ax.set_xlabel("Spatial frequency (cycles/pixel)")
    ax.set_ylabel("Foggy capture / archive PSD")
    ax.set_title("Paired image-domain PSD retention")
    ax.grid(True, which="both", axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    text = (
        f"low: {summary['low_band_median_pair_ratio']:.4f}\n"
        f"high: {summary['high_band_median_pair_ratio']:.4f}\n"
        f"low/high: {summary['low_over_high_relative_suppression']:.1f}x"
    )
    ax.text(
        0.035,
        0.04,
        text,
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9, "pad": 4},
    )
    fig.tight_layout()
    fig.savefig(out_dir / "paired_psd_retention.png", dpi=300)
    fig.savefig(out_dir / "paired_psd_retention.pdf")
    plt.close(fig)


def write_section(summary: dict, out_dir: Path) -> None:
    low_ci = summary["low_band_median_pair_ratio_ci95"]
    high_ci = summary["high_band_median_pair_ratio_ci95"]
    text = (
        r"""\section{Image-Domain Power-Spectral Analysis}
\label{secS:psd_image_domain}

"""
        + f"We used the paired chamber data to quantify how fog changes spatial-frequency content in the recorded images. For each held-out pair, the foggy chamber capture and the corresponding archive image displayed on the screen were converted to luminance, resized to ${summary['image_size']} \\times {summary['image_size']}$, mean-subtracted, Hann-windowed, Fourier transformed, and radially averaged into {summary['radial_bins']} spatial-frequency bins. We then computed the per-pair radial power-spectral-density (PSD) ratio,\n"
        + r"""\[
R_{\mathrm{PSD}}(f) =
\frac{P_{\mathrm{fog}}(f)}
     {P_{\mathrm{archive}}(f)} ,
\]
"""
        + f"where $P_{{\\mathrm{{fog}}}}(f)$ is the foggy camera-capture PSD and $P_{{\\mathrm{{archive}}}}(f)$ is the PSD of the displayed archive image. The deterministic every-tenth held-out split gave $n={summary['held_out_pairs']}$ pairs.\n\n"
        + "This is an image-domain degradation measurement, not a physical transmission spectrum of the fog. The ratio includes the display, chamber, fog, lens, sensor, exposure, camera processing, and resizing pipeline. It is therefore used only to summarize loss of recorded spatial detail.\n\n"
        + f"The median paired PSD ratio was {summary['low_band_median_pair_ratio']:.4f} (95\\% bootstrap CI {low_ci[0]:.4f}--{low_ci[1]:.4f}) over low spatial frequencies, 0.01--0.05 cycles/pixel, and {summary['high_band_median_pair_ratio']:.4f} (95\\% CI {high_ci[0]:.4f}--{high_ci[1]:.4f}) over higher spatial frequencies, 0.15--0.24 cycles/pixel. Thus high-frequency power was retained {summary['low_over_high_relative_suppression']:.1f}$\\times$ less than low-frequency power in the paired image-domain measurement. This frequency-domain result supports the Laplacian-retention analysis in the main text: foggy captures lose fine spatial structure disproportionately, making high-texture scenes harder to restore.\n\n"
        + r"""\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\textwidth]{figures/supplement/paired_psd_retention.pdf}
\caption{Paired image-domain power-spectral-density retention for the held-out chamber split. The solid line shows the median foggy-capture/archive PSD ratio across image pairs; the shaded band shows the interquartile range. The dashed line shows the ratio of mean foggy and archive PSDs. Green and orange bands mark the low- and high-frequency ranges summarized in the text.}
\label{figS:paired_psd_retention}
\end{figure}
"""
    )
    (out_dir / "supplement_psd_section.tex").write_text(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fog-root", type=Path, default=DEFAULT_FOG_ROOT)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--bins", type=int, default=96)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = resolve_pairs(args.fog_root, args.archive_root)
    per_pair_df, profile_df, summary = analyze(records, args.size, args.bins, args.out_dir)
    make_figure(profile_df, summary, args.out_dir)
    write_section(summary, args.out_dir)
    print(json.dumps({
        "all_pairs": len(records),
        "held_out_pairs": len(per_pair_df),
        "low_band_median_pair_ratio": summary["low_band_median_pair_ratio"],
        "high_band_median_pair_ratio": summary["high_band_median_pair_ratio"],
        "low_over_high_relative_suppression": summary["low_over_high_relative_suppression"],
        "out_dir": str(args.out_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
