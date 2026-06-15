#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy import ndimage, stats


ROOT = Path(".")
OUT = ROOT / "metric_to_psnr"
FIG = OUT / "figures"
TAB = OUT / "tables"
PAIRED_METRICS = ROOT / "outputs_20260605" / "paired_blur_stats" / "paired_blur_psf_equivalent_metrics.csv"
PAIRED_CORR = ROOT / "outputs_20260605" / "paired_blur_stats" / "paired_blur_psf_equivalent_correlations.csv"
OLD_CORR = ROOT / "outputs_20260605" / "tables" / "fog_statistic_psnr_correlations.csv"
BASE_TABLE = ROOT / "outputs_20260605" / "tables" / "test_physics_plus_nafnet_metrics.csv"


LABELS = {
    "laplacian_step1_mean_abs_ratio": "Laplacian mean abs ratio, step 1",
    "laplacian_step2_mean_abs_ratio": "Laplacian mean abs ratio, step 2",
    "laplacian_variance_ratio": "Laplacian variance ratio",
    "gradient_step1_mean_abs_ratio": "Gradient mean abs ratio, step 1",
    "gradient_ratio_recomputed": "Gradient magnitude ratio",
    "fog_dark_channel_q90": "Foggy dark-channel q90",
    "fog_dark_channel_q95": "Foggy dark-channel q95",
    "fog_luminance_q75": "Foggy luminance q75",
    "input_psnr": "Input PSNR",
    "equiv_gaussian_fwhm_px": "Equivalent FWHM",
}


COLORS = {
    "apparel": "#4267ac",
    "artwork": "#b64545",
    "cars": "#4f8a57",
    "dishes": "#c17d20",
    "furniture": "#7359a6",
    "illustrations": "#5b8d9a",
}


def read_gray(path: str | Path) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def robust_img(x: np.ndarray, lo=1, hi=99) -> np.ndarray:
    a, b = np.percentile(x, [lo, hi])
    if b <= a:
        return np.zeros_like(x)
    return np.clip((x - a) / (b - a), 0, 1)


def pretty_metric(name: str) -> str:
    return LABELS.get(name, name.replace("_", " "))


def save_tables() -> None:
    paired_corr = pd.read_csv(PAIRED_CORR)
    old_corr = pd.read_csv(OLD_CORR)
    top_paired = paired_corr.head(12).copy()
    top_input = old_corr[old_corr["metric_group"] == "input_only"].head(8).copy()
    top_paired["metric"] = top_paired["column"].map(pretty_metric)
    top_input["metric"] = top_input["column"].map(pretty_metric)
    top_paired[["metric", "column", "spearman_rho", "category_centered_spearman_rho"]].to_csv(
        TAB / "top_paired_metrics.csv", index=False
    )
    top_input[["metric", "column", "spearman_rho", "category_centered_spearman_rho"]].to_csv(
        TAB / "top_unpaired_metrics.csv", index=False
    )


def plot_paired_vs_unpaired() -> None:
    paired_corr = pd.read_csv(PAIRED_CORR)
    old_corr = pd.read_csv(OLD_CORR)
    rows = []
    for _, r in paired_corr.head(6).iterrows():
        rows.append(("paired", pretty_metric(r["column"]), r["spearman_rho"], r["category_centered_spearman_rho"]))
    for _, r in old_corr[old_corr["metric_group"] == "input_only"].head(6).iterrows():
        rows.append(("foggy-only", pretty_metric(r["column"]), r["spearman_rho"], r["category_centered_spearman_rho"]))
    df = pd.DataFrame(rows, columns=["group", "metric", "rho", "cc_rho"])
    df = df.sort_values("rho", ascending=True)
    colors = ["#315c9b" if g == "paired" else "#999999" for g in df["group"]]
    fig, ax = plt.subplots(figsize=(7.3, 4.9))
    ax.barh(df["metric"], df["rho"], color=colors)
    ax.scatter(df["cc_rho"], df["metric"], color="black", s=24, label="category-centered rho", zorder=3)
    ax.set_xlabel("Spearman correlation with NAFNet PSNR")
    ax.set_xlim(0, 0.70)
    ax.grid(axis="x", color="0.88", linewidth=0.8)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Paired metrics outperform foggy-image-only proxies")
    fig.tight_layout()
    fig.savefig(FIG / "paired_vs_unpaired_correlation_rank.png", dpi=240)
    plt.close(fig)


def plot_scale_sweep() -> None:
    corr = pd.read_csv(PAIRED_CORR).set_index("column")
    steps = np.array([1, 2, 3, 4, 6, 8, 12])
    sigmas = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

    def vals(prefix: str, xs) -> tuple[list[float], list[float]]:
        rho = []
        cc = []
        for x in xs:
            key = prefix.format(str(x).replace(".", "p"))
            rho.append(float(corr.loc[key, "spearman_rho"]))
            cc.append(float(corr.loc[key, "category_centered_spearman_rho"]))
        return rho, cc

    lap_mean, lap_mean_cc = vals("laplacian_step{}_mean_abs_ratio", steps)
    lap_var, lap_var_cc = vals("laplacian_step{}_variance_ratio", steps)
    grad_mean, grad_mean_cc = vals("gradient_step{}_mean_abs_ratio", steps)
    log_mean, log_mean_cc = vals("log_sigma{}_mean_abs_ratio", sigmas)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), sharey=True)
    axes[0].plot(steps, lap_mean, marker="o", label="Laplacian mean abs")
    axes[0].plot(steps, lap_var, marker="o", label="Laplacian variance")
    axes[0].plot(steps, grad_mean, marker="o", label="Gradient mean abs")
    axes[0].set_xlabel("finite-difference step (px)")
    axes[0].set_ylabel("Spearman rho")
    axes[0].set_title("Finite-difference scale")
    axes[0].grid(color="0.88", linewidth=0.8)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(sigmas, log_mean, marker="o", color="#b64545", label="LoG mean abs")
    axes[1].plot(sigmas, log_mean_cc, marker="s", color="#b64545", alpha=0.55, label="category-centered")
    axes[1].set_xlabel("Gaussian sigma (px)")
    axes[1].set_title("Laplacian-of-Gaussian scale")
    axes[1].grid(color="0.88", linewidth=0.8)
    axes[1].legend(frameon=False, fontsize=8)
    axes[0].set_ylim(0, 0.68)
    fig.tight_layout()
    fig.savefig(FIG / "laplacian_gradient_scale_sweep.png", dpi=240)
    plt.close(fig)


def plot_best_scatter() -> None:
    df = pd.read_csv(PAIRED_METRICS)
    xcol = "laplacian_step1_mean_abs_ratio"
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    for cat, sub in df.groupby("category"):
        ax.scatter(sub[xcol], sub["nafnet_psnr"], s=22, alpha=0.72, color=COLORS.get(cat), label=cat)
    x = df[xcol].to_numpy()
    y = df["nafnet_psnr"].to_numpy()
    slope, intercept, r, _, _ = stats.linregress(x, y)
    xx = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    ax.plot(xx, intercept + slope * xx, color="black", linewidth=1.2)
    rho = stats.spearmanr(x, y).statistic
    ax.set_xscale("log")
    ax.set_xlabel("Laplacian mean absolute ratio, step 1")
    ax.set_ylabel("NAFNet PSNR (dB)")
    ax.set_title(f"Best paired statistic: Spearman rho = {rho:.3f}")
    ax.grid(color="0.88", linewidth=0.8)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "best_metric_vs_psnr.png", dpi=240)
    plt.close(fig)


def plot_examples() -> None:
    base = pd.read_csv(BASE_TABLE)
    df = pd.read_csv(PAIRED_METRICS)
    merged = df.merge(base[["category", "image_name", "fog_path", "clear_path"]], on=["category", "image_name"], how="left")
    xcol = "laplacian_step1_mean_abs_ratio"
    picks = [
        ("low metric / low PSNR", merged.sort_values(xcol).iloc[12]),
        ("high metric / high PSNR", merged.sort_values(xcol).iloc[-12]),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 3.9))
    for row_idx, (label, row) in enumerate(picks):
        clear = read_gray(row["clear_path"])
        fog = read_gray(row["fog_path"])
        clear_lap = np.abs(ndimage.laplace(clear))
        fog_lap = np.abs(ndimage.laplace(fog))
        items = [
            ("clear", clear),
            ("fog", fog),
            ("|Lap clear|", clear_lap),
            ("|Lap fog|", fog_lap),
        ]
        for col_idx, (title, arr) in enumerate(items):
            ax = axes[row_idx, col_idx]
            ax.imshow(robust_img(arr), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(title, fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(
                    f"{label}\nratio={row[xcol]:.3g}\nPSNR={row['nafnet_psnr']:.1f} dB",
                    fontsize=8,
                )
    fig.suptitle("The selected metric measures retained high-frequency structure", y=0.99, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "clear_fog_laplacian_examples.png", dpi=240)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    save_tables()
    plot_paired_vs_unpaired()
    plot_scale_sweep()
    plot_best_scatter()
    plot_examples()


if __name__ == "__main__":
    main()
