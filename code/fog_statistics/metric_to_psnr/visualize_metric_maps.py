#!/usr/bin/env python3
"""Render 2D heatmaps for Laplacian/gradient paired fog metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy import ndimage


ROOT = Path(".")
TABLE = ROOT / "metric_to_psnr" / "deep_dive_20260605" / "all_metric_features.csv"
OUTDIR = ROOT / "metric_to_psnr" / "deep_dive_20260605" / "metric_map_figures"
SIZE = 512
EPS = 1e-6


def read_rgb(path: str | Path) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if img.size != (SIZE, SIZE):
        img = img.resize((SIZE, SIZE), Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def luma(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def gradient(gray: np.ndarray) -> np.ndarray:
    gx = ndimage.sobel(gray, axis=1, mode="reflect") / 8.0
    gy = ndimage.sobel(gray, axis=0, mode="reflect") / 8.0
    return np.sqrt(gx * gx + gy * gy)


def local_var(x: np.ndarray, radius: int = 7) -> np.ndarray:
    size = 2 * radius + 1
    mean = ndimage.uniform_filter(x, size=size, mode="reflect")
    mean_sq = ndimage.uniform_filter(x * x, size=size, mode="reflect")
    return np.maximum(mean_sq - mean * mean, 0.0)


def robust_limits(x: np.ndarray, lo: float = 1, hi: float = 99) -> tuple[float, float]:
    vals = x[np.isfinite(x)]
    if vals.size == 0:
        return 0.0, 1.0
    a, b = np.percentile(vals, [lo, hi])
    if b <= a:
        b = a + 1e-6
    return float(a), float(b)


def choose_examples(df: pd.DataFrame) -> pd.DataFrame:
    score = "laplacian_step1_mean_abs_ratio"
    low = df.sort_values(score).iloc[0]
    high_pool = df[df["nafnet_psnr"] > df["nafnet_psnr"].quantile(0.85)]
    high = high_pool.sort_values(score).iloc[-1] if len(high_pool) else df.sort_values(score).iloc[-1]
    median_idx = (df[score] - df[score].median()).abs().idxmin()
    mid = df.loc[median_idx]
    return pd.DataFrame([low, mid, high])


def choose_category_examples(df: pd.DataFrame, category: str, n: int, exclude: set[str] | None = None) -> pd.DataFrame:
    score = "laplacian_step1_mean_abs_ratio"
    exclude = exclude or set()
    sub = df[(df["category"] == category) & (~df["image_name"].isin(exclude))].copy()
    if len(sub) < n:
        raise ValueError(f"Only found {len(sub)} rows for category={category!r}; need {n}.")
    quantiles = np.linspace(0.10, 0.90, n)
    chosen = []
    used: set[int] = set()
    for q in quantiles:
        target = sub[score].quantile(float(q))
        candidates = (sub[score] - target).abs().sort_values()
        for idx in candidates.index:
            if int(idx) not in used:
                chosen.append(sub.loc[idx])
                used.add(int(idx))
                break
    return pd.DataFrame(chosen).sort_values(score)


def add_heatmap(ax, arr: np.ndarray, title: str, cmap: str, *, log_ratio: bool = False) -> None:
    if log_ratio:
        vmin, vmax = -4.0, 2.0
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        vmin, vmax = robust_limits(arr)
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=7)


def mark_best_axis(ax) -> None:
    ax.set_title("BEST SINGLE PREDICTOR\nlog2 |Lap fog| / |Lap clear|", fontsize=9, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(3.0)
        spine.set_edgecolor("#ffd23f")


def metric_maps(clear_rgb: np.ndarray, fog_rgb: np.ndarray) -> dict[str, np.ndarray]:
    clear = luma(clear_rgb)
    fog = luma(fog_rgb)
    lap_clear = ndimage.laplace(clear, mode="reflect")
    lap_fog = ndimage.laplace(fog, mode="reflect")
    abs_lap_clear = np.abs(lap_clear)
    abs_lap_fog = np.abs(lap_fog)
    var_lap_clear = local_var(lap_clear)
    var_lap_fog = local_var(lap_fog)
    grad_clear = gradient(clear)
    grad_fog = gradient(fog)
    return {
        "abs_lap_clear": abs_lap_clear,
        "abs_lap_fog": abs_lap_fog,
        "var_lap_clear": var_lap_clear,
        "var_lap_fog": var_lap_fog,
        "log2_abs_lap_ratio": np.log2((abs_lap_fog + EPS) / (abs_lap_clear + EPS)),
        "log2_var_lap_ratio": np.log2((var_lap_fog + EPS * EPS) / (var_lap_clear + EPS * EPS)),
        "grad_clear": grad_clear,
        "grad_fog": grad_fog,
        "log2_grad_ratio": np.log2((grad_fog + EPS) / (grad_clear + EPS)),
    }


def render_one(row: pd.Series, outdir: Path = OUTDIR) -> Path:
    clear_rgb = read_rgb(row["clear_path"])
    fog_rgb = read_rgb(row["fog_path"])
    maps = metric_maps(clear_rgb, fog_rgb)
    label = f"{row['category']}_{row['image_name']}".replace("/", "_").replace(".", "_")
    out = outdir / f"metric_maps_{label}.png"

    fig, axes = plt.subplots(3, 4, figsize=(13.0, 9.0))
    axes = axes.ravel()
    axes[0].imshow(clear_rgb)
    axes[0].set_title("clear RGB", fontsize=9)
    axes[1].imshow(fog_rgb)
    axes[1].set_title("fog RGB", fontsize=9)
    for ax in axes[:2]:
        ax.set_xticks([])
        ax.set_yticks([])

    add_heatmap(axes[2], maps["abs_lap_clear"], "|Laplacian| clear", "magma")
    add_heatmap(axes[3], maps["abs_lap_fog"], "|Laplacian| fog", "magma")
    add_heatmap(axes[4], maps["var_lap_clear"], "local Var(Laplacian) clear", "viridis")
    add_heatmap(axes[5], maps["var_lap_fog"], "local Var(Laplacian) fog", "viridis")
    add_heatmap(axes[6], maps["log2_abs_lap_ratio"], "log2 |Lap fog| / |Lap clear|", "coolwarm", log_ratio=True)
    mark_best_axis(axes[6])
    add_heatmap(axes[7], maps["log2_var_lap_ratio"], "log2 VarLap fog / clear", "coolwarm", log_ratio=True)
    add_heatmap(axes[8], maps["grad_clear"], "gradient magnitude clear", "cividis")
    add_heatmap(axes[9], maps["grad_fog"], "gradient magnitude fog", "cividis")
    add_heatmap(axes[10], maps["log2_grad_ratio"], "log2 gradient fog / clear", "coolwarm", log_ratio=True)
    axes[11].axis("off")
    axes[11].text(
        0.02,
        0.95,
        "\n".join(
            [
                f"{row['category']} / {row['image_name']}",
                f"NAFNet PSNR: {row['nafnet_psnr']:.2f} dB",
                f"Laplacian ratio: {row['laplacian_step1_mean_abs_ratio']:.3f}",
                f"GMS std: {row['gms_std']:.3f}",
                "",
                "Ratio heatmaps use log2 scale:",
                "blue = fog has less structure",
                "white = roughly unchanged",
                "red = fog has more local response",
            ]
        ),
        ha="left",
        va="top",
        fontsize=10,
    )
    fig.suptitle("Full-color 2D Laplacian and gradient metric maps", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def render_overview(
    rows: pd.DataFrame,
    out_name: str = "metric_ratio_maps_low_mid_high_overview.png",
    title: str = "Low, middle, and high paired metric examples",
    outdir: Path = OUTDIR,
) -> Path:
    out = outdir / out_name
    fig_height = max(6.0, 2.7 * len(rows))
    fig, axes = plt.subplots(len(rows), 5, figsize=(14, fig_height), squeeze=False)
    for r, (_, row) in enumerate(rows.iterrows()):
        clear_rgb = read_rgb(row["clear_path"])
        fog_rgb = read_rgb(row["fog_path"])
        maps = metric_maps(clear_rgb, fog_rgb)
        axes[r, 0].imshow(fog_rgb)
        axes[r, 0].set_title("fog RGB" if r == 0 else "", fontsize=9)
        axes[r, 0].set_ylabel(
            f"{row['category']}\nPSNR {row['nafnet_psnr']:.1f} dB\nLap ratio {row['laplacian_step1_mean_abs_ratio']:.3f}",
            fontsize=8,
        )
        add_heatmap(axes[r, 1], maps["log2_abs_lap_ratio"], "log2 abs Lap ratio" if r == 0 else "", "coolwarm", log_ratio=True)
        mark_best_axis(axes[r, 1])
        add_heatmap(axes[r, 2], maps["log2_var_lap_ratio"], "log2 VarLap ratio" if r == 0 else "", "coolwarm", log_ratio=True)
        add_heatmap(axes[r, 3], maps["log2_grad_ratio"], "log2 gradient ratio" if r == 0 else "", "coolwarm", log_ratio=True)
        add_heatmap(axes[r, 4], maps["grad_fog"], "fog gradient" if r == 0 else "", "cividis")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(TABLE)
    examples = choose_examples(df)
    paths = [render_one(row) for _, row in examples.iterrows()]
    paths.append(render_overview(examples))

    dishes_dir = OUTDIR / "dishes_five"
    dishes_dir.mkdir(parents=True, exist_ok=True)
    dishes = choose_category_examples(df, "dishes", 5, exclude={"image0230.jpeg"})
    dishes_paths = [render_one(row, dishes_dir) for _, row in dishes.iterrows()]
    dishes_paths.append(
        render_overview(
            dishes,
            out_name="dishes_five_laplacian_gradient_overview.png",
            title="Five dishes examples across the Laplacian-ratio range",
            outdir=dishes_dir,
        )
    )
    dishes_manifest = dishes_dir / "dishes_five_manifest.txt"
    dishes_manifest.write_text("\n".join(str(p) for p in dishes_paths) + "\n", encoding="utf-8")
    paths.extend(dishes_paths)

    manifest = OUTDIR / "metric_map_manifest.txt"
    manifest.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")
    print(manifest)


if __name__ == "__main__":
    main()
