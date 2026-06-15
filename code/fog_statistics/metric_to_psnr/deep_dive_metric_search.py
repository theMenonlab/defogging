#!/usr/bin/env python3
"""Deep search for fog-image metrics that predict NAFNet PSNR.

The script keeps two kinds of scores separate:
1. Single, reportable image-pair metrics.
2. Composite predictors evaluated with out-of-fold predictions.

The composite models are useful as an upper-bound check, but the single-metric
table is the safer source for physically interpretable fog statistics.
"""

from __future__ import annotations

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
from scipy import ndimage, stats
from skimage import color, feature, metrics, transform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR


ROOT = Path(".")
BASE_TABLE = ROOT / "outputs_20260605" / "tables" / "test_physics_plus_nafnet_metrics.csv"
BLUR_TABLE = ROOT / "outputs_20260605" / "paired_blur_stats" / "paired_blur_psf_equivalent_metrics.csv"
OUTDIR = ROOT / "metric_to_psnr" / "deep_dive_20260605"
FIGDIR = OUTDIR / "figures"
SIZE = 512
EPS = 1e-12


def read_rgb(path: str | Path) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if img.size != (SIZE, SIZE):
        img = img.resize((SIZE, SIZE), Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def ratio(num: float, den: float) -> float:
    return float(num / max(den, EPS))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    finite = np.isfinite(a) & np.isfinite(b)
    if int(finite.sum()) < 3:
        return float("nan")
    if np.nanstd(a[finite]) < EPS or np.nanstd(b[finite]) < EPS:
        return float("nan")
    return float(np.corrcoef(a[finite], b[finite])[0, 1])


def gradient(gray: np.ndarray) -> np.ndarray:
    gx = ndimage.sobel(gray, axis=1, mode="reflect") / 8.0
    gy = ndimage.sobel(gray, axis=0, mode="reflect") / 8.0
    return np.sqrt(gx * gx + gy * gy)


def laplacian_abs(gray: np.ndarray) -> np.ndarray:
    return np.abs(ndimage.laplace(gray, mode="reflect"))


def radial_power(gray: np.ndarray, bins: int = 48) -> tuple[np.ndarray, np.ndarray]:
    h, w = gray.shape
    win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    spec = np.fft.fftshift(np.fft.fft2((gray - float(gray.mean())) * win))
    power = np.abs(spec) ** 2
    yy, xx = np.indices((h, w), dtype=np.float32)
    rr = np.sqrt((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2)
    edges = np.linspace(0.0, min(h, w) / 2.0, bins + 1)
    which = np.digitize(rr.reshape(-1), edges) - 1
    sums = np.bincount(which, weights=power.reshape(-1), minlength=bins)
    counts = np.bincount(which, minlength=bins)
    freq = 0.5 * (edges[:-1] + edges[1:]) / min(h, w)
    return freq, sums[:bins] / np.maximum(counts[:bins], 1)


def affine_match_residual(clear: np.ndarray, fog: np.ndarray) -> dict[str, float]:
    x = clear.reshape(-1).astype(np.float64)
    y = fog.reshape(-1).astype(np.float64)
    a, b = np.polyfit(x, y, 1)
    pred = a * clear + b
    resid = fog - pred
    return {
        "affine_luma_slope": float(a),
        "affine_luma_intercept": float(b),
        "affine_luma_rmse": float(np.sqrt(np.mean(resid * resid))),
        "affine_luma_mae": float(np.mean(np.abs(resid))),
        "affine_luma_resid_q90": float(np.quantile(np.abs(resid), 0.90)),
    }


def multiscale_ssim(clear: np.ndarray, fog: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    vals: list[float] = []
    c = clear
    f = fog
    for level in range(4):
        key = f"ssim_gray_scale{2 ** level}"
        val = metrics.structural_similarity(c, f, data_range=1.0)
        out[key] = float(val)
        vals.append(float(max(val, 1e-6)))
        if level < 3:
            c = transform.downscale_local_mean(c, (2, 2))
            f = transform.downscale_local_mean(f, (2, 2))
    out["ssim_gray_multiscale_geom"] = float(np.exp(np.mean(np.log(vals))))
    out["ssim_gray_multiscale_mean"] = float(np.mean(vals))
    return out


def block_stats(clear: np.ndarray, fog: np.ndarray, block: int) -> dict[str, float]:
    h = clear.shape[0] // block * block
    w = clear.shape[1] // block * block
    c = clear[:h, :w].reshape(h // block, block, w // block, block)
    f = fog[:h, :w].reshape(h // block, block, w // block, block)
    c_blocks = c.mean(axis=(1, 3))
    f_blocks = f.mean(axis=(1, 3))
    c_std = c.std(axis=(1, 3))
    f_std = f.std(axis=(1, 3))
    mean_abs = np.abs(f_blocks - c_blocks)
    contrast_ratio = f_std / np.maximum(c_std, EPS)
    prefix = f"block{block}"
    return {
        f"{prefix}_mean_abs_luma_delta_mean": float(np.mean(mean_abs)),
        f"{prefix}_mean_abs_luma_delta_q90": float(np.quantile(mean_abs, 0.90)),
        f"{prefix}_contrast_ratio_mean": float(np.mean(contrast_ratio)),
        f"{prefix}_contrast_ratio_q10": float(np.quantile(contrast_ratio, 0.10)),
        f"{prefix}_contrast_ratio_q50": float(np.quantile(contrast_ratio, 0.50)),
        f"{prefix}_contrast_ratio_q90": float(np.quantile(contrast_ratio, 0.90)),
    }


def edge_and_texture_metrics(clear: np.ndarray, fog: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    cg = gradient(clear)
    fg = gradient(fog)
    cl = laplacian_abs(clear)
    fl = laplacian_abs(fog)

    gms = (2.0 * cg * fg + 0.0026) / (cg * cg + fg * fg + 0.0026)
    out["gms_mean"] = float(np.mean(gms))
    out["gms_std"] = float(np.std(gms))
    out["gms_q10"] = float(np.quantile(gms, 0.10))
    out["gms_q50"] = float(np.quantile(gms, 0.50))
    out["gms_q90"] = float(np.quantile(gms, 0.90))

    for pct in (50, 70, 80, 90, 95):
        g_thr = float(np.percentile(cg, pct))
        l_thr = float(np.percentile(cl, pct))
        gmask = cg >= g_thr
        lmask = cl >= l_thr
        out[f"clear_grad_p{pct}_fog_grad_ratio"] = ratio(float(fg[gmask].mean()), float(cg[gmask].mean()))
        out[f"clear_grad_p{pct}_fog_grad_energy_ratio"] = ratio(
            float(np.mean(fg[gmask] * fg[gmask])), float(np.mean(cg[gmask] * cg[gmask]))
        )
        out[f"clear_grad_p{pct}_edge_recall_fixed"] = float(np.mean(fg[gmask] >= g_thr))
        out[f"clear_lap_p{pct}_fog_lap_ratio"] = ratio(float(fl[lmask].mean()), float(cl[lmask].mean()))
        out[f"clear_lap_p{pct}_lap_recall_fixed"] = float(np.mean(fl[lmask] >= l_thr))

    clear_edges = feature.canny(clear, sigma=1.0, low_threshold=0.03, high_threshold=0.09)
    fog_edges = feature.canny(fog, sigma=1.0, low_threshold=0.03, high_threshold=0.09)
    inter = np.logical_and(clear_edges, fog_edges).sum()
    union = np.logical_or(clear_edges, fog_edges).sum()
    out["canny_edge_jaccard_fixed"] = float(inter / max(union, 1))
    out["canny_clear_edge_recall_fixed"] = float(inter / max(clear_edges.sum(), 1))
    out["canny_fog_edge_density_fixed"] = float(fog_edges.mean())
    out["canny_clear_edge_density_fixed"] = float(clear_edges.mean())

    low_texture = cg <= np.percentile(cg, 25)
    out["low_clear_texture_fog_luma_std"] = float(np.std(fog[low_texture]))
    out["low_clear_texture_abs_delta_mean"] = float(np.mean(np.abs(fog[low_texture] - clear[low_texture])))
    return out


def spectral_metrics(clear: np.ndarray, fog: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    c_z = (clear - clear.mean()) / max(clear.std(), EPS)
    f_z = (fog - fog.mean()) / max(fog.std(), EPS)
    freq, cp = radial_power(c_z)
    _, fp = radial_power(f_z)
    bands = {
        "vlow": (0.00, 0.035),
        "low": (0.035, 0.08),
        "mid": (0.08, 0.16),
        "high": (0.16, 0.27),
        "vhigh": (0.27, 0.50),
    }
    band_ratios = {}
    for name, (lo, hi) in bands.items():
        mask = (freq >= lo) & (freq < hi)
        r = ratio(float(fp[mask].sum()), float(cp[mask].sum()))
        out[f"zpsd_{name}_ratio"] = r
        band_ratios[name] = r
    out["zpsd_high_low_ratio"] = ratio(band_ratios["high"], band_ratios["low"])
    out["zpsd_vhigh_low_ratio"] = ratio(band_ratios["vhigh"], band_ratios["low"])
    return out


def color_metrics(clear_rgb: np.ndarray, fog_rgb: np.ndarray) -> dict[str, float]:
    clear_lab = color.rgb2lab(clear_rgb)
    fog_lab = color.rgb2lab(fog_rgb)
    de = np.sqrt(np.sum((fog_lab - clear_lab) ** 2, axis=-1))
    clear_hsv = color.rgb2hsv(clear_rgb)
    fog_hsv = color.rgb2hsv(fog_rgb)
    return {
        "deltae_mean": float(de.mean()),
        "deltae_std": float(de.std()),
        "deltae_q50": float(np.quantile(de, 0.50)),
        "deltae_q90": float(np.quantile(de, 0.90)),
        "saturation_ratio": ratio(float(fog_hsv[..., 1].mean()), float(clear_hsv[..., 1].mean())),
        "saturation_delta_mean": float(fog_hsv[..., 1].mean() - clear_hsv[..., 1].mean()),
    }


def pair_metrics(row: pd.Series) -> dict[str, float | str]:
    clear_rgb = read_rgb(row["clear_path"])
    fog_rgb = read_rgb(row["fog_path"])
    clear = luminance(clear_rgb)
    fog = luminance(fog_rgb)
    clear_z = (clear - clear.mean()) / max(clear.std(), EPS)
    fog_z = (fog - fog.mean()) / max(fog.std(), EPS)
    resid = fog_z - clear_z

    out: dict[str, float | str] = {
        "category": row["category"],
        "image_name": row["image_name"],
        "nafnet_psnr": float(row["nafnet_psnr"]),
        "pair_luma_mae": float(np.mean(np.abs(fog - clear))),
        "pair_luma_rmse": float(np.sqrt(np.mean((fog - clear) ** 2))),
        "pair_luma_corr": safe_corr(clear.reshape(-1), fog.reshape(-1)),
        "zscore_luma_rmse": float(np.sqrt(np.mean(resid * resid))),
        "zscore_luma_mae": float(np.mean(np.abs(resid))),
        "zscore_luma_corr": safe_corr(clear_z.reshape(-1), fog_z.reshape(-1)),
        "rgb_ssim": float(metrics.structural_similarity(clear_rgb, fog_rgb, channel_axis=-1, data_range=1.0)),
        "rgb_psnr": float(metrics.peak_signal_noise_ratio(clear_rgb, fog_rgb, data_range=1.0)),
        "rgb_nrmse": float(metrics.normalized_root_mse(clear_rgb, fog_rgb)),
    }
    out.update(affine_match_residual(clear, fog))
    out.update(multiscale_ssim(clear, fog))
    out.update(edge_and_texture_metrics(clear, fog))
    out.update(spectral_metrics(clear, fog))
    out.update(color_metrics(clear_rgb, fog_rgb))
    for block in (16, 32, 64):
        out.update(block_stats(clear, fog, block))
    return out


def compute_rich_metrics(base: pd.DataFrame) -> pd.DataFrame:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTDIR / "rich_paired_metrics.csv"
    if out_path.exists():
        return pd.read_csv(out_path)
    rows = []
    for idx, (_, row) in enumerate(base.iterrows(), start=1):
        if idx % 25 == 0:
            print(f"computed rich pair metrics for {idx}/{len(base)} images", flush=True)
        rows.append(pair_metrics(row))
    rich = pd.DataFrame(rows)
    rich.to_csv(out_path, index=False)
    return rich


def merge_features(base: pd.DataFrame, blur: pd.DataFrame, rich: pd.DataFrame) -> pd.DataFrame:
    blur_keep = blur.drop(columns=[c for c in ("nafnet_psnr", "input_psnr") if c in blur.columns])
    rich_keep = rich.drop(columns=["nafnet_psnr"])
    df = base.merge(blur_keep, on=["category", "image_name"], how="left")
    df = df.merge(rich_keep, on=["category", "image_name"], how="left", suffixes=("", "_rich"))
    return df


def single_metric_correlations(df: pd.DataFrame) -> pd.DataFrame:
    y = df["nafnet_psnr"].to_numpy(dtype=float)
    y_center = df["nafnet_psnr"] - df.groupby("category")["nafnet_psnr"].transform("mean")
    rows = []
    excluded = {
        "nafnet_psnr",
        "nafnet_ssim",
        "nafnet_mae",
        "nafnet_mse",
    }
    for col in df.columns:
        if col in excluded or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        x = df[col].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if int(finite.sum()) < 25 or np.nanstd(x[finite]) < EPS:
            continue
        xc = pd.Series(df[col]).astype(float) - df.groupby("category")[col].transform("mean")
        finite_c = np.isfinite(xc) & np.isfinite(y_center)
        rows.append(
            {
                "column": col,
                "n": int(finite.sum()),
                "spearman_rho": float(stats.spearmanr(x[finite], y[finite]).statistic),
                "pearson_r": float(stats.pearsonr(x[finite], y[finite]).statistic),
                "category_centered_spearman_rho": float(
                    stats.spearmanr(np.asarray(xc)[finite_c], np.asarray(y_center)[finite_c]).statistic
                ),
                "abs_spearman_rho": abs(float(stats.spearmanr(x[finite], y[finite]).statistic)),
            }
        )
    corr = pd.DataFrame(rows).sort_values("abs_spearman_rho", ascending=False)
    corr.to_csv(OUTDIR / "single_metric_correlations.csv", index=False)
    return corr


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    include_category: bool


def numeric_feature_columns(df: pd.DataFrame, mode: str) -> list[str]:
    blocked = {
        "nafnet_psnr",
        "nafnet_ssim",
        "nafnet_mae",
        "nafnet_mse",
    }
    meta = {"split", "category", "image_name", "fog_path", "clear_path"}
    cols = []
    for col in df.columns:
        if col in blocked or col in meta:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if mode == "paired_only":
            keep = (
                col in {"input_psnr", "mi_bits", "dark_channel_delta", "rms_contrast_ratio", "luminance_std_ratio",
                        "gradient_ratio", "mean_intensity_shift"}
                or col.startswith(("t_", "tau_", "dcp_t_", "gradient_", "laplacian_", "tenengrad_", "brenner_",
                                   "psd_", "zscore_", "clear_fog_", "equiv_", "gaussian_", "log_", "pair_",
                                   "affine_", "ssim_", "rgb_", "gms_", "clear_grad_", "clear_lap_", "canny_",
                                   "low_clear_", "zpsd_", "deltae_", "saturation_", "block"))
            )
            if not keep:
                continue
        elif mode == "foggy_only":
            if not col.startswith("fog_"):
                continue
        cols.append(col)
    return cols


def preprocess(numeric_cols: list[str], include_category: bool, scale: bool, select_k: int | None = None) -> Pipeline:
    num_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    if select_k is not None:
        num_steps.append(("select", SelectKBest(score_func=f_regression, k=min(select_k, len(numeric_cols)))))
    transformers: list[tuple[str, object, list[str]]] = [("num", Pipeline(num_steps), numeric_cols)]
    if include_category:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), ["category"]))
    return ColumnTransformer(transformers=transformers)


def model_specs(numeric_cols: list[str], include_category: bool) -> list[ModelSpec]:
    ridge_alphas = np.logspace(-3, 4, 30)
    specs = [
        ModelSpec(
            f"ridge_all{'_cat' if include_category else ''}",
            Pipeline(
                [
                    ("prep", preprocess(numeric_cols, include_category, scale=True)),
                    ("model", RidgeCV(alphas=ridge_alphas)),
                ]
            ),
            include_category,
        ),
        ModelSpec(
            f"ridge_select30{'_cat' if include_category else ''}",
            Pipeline(
                [
                    ("prep", preprocess(numeric_cols, include_category, scale=True, select_k=30)),
                    ("model", RidgeCV(alphas=ridge_alphas)),
                ]
            ),
            include_category,
        ),
        ModelSpec(
            f"svr_rbf_select40{'_cat' if include_category else ''}",
            Pipeline(
                [
                    ("prep", preprocess(numeric_cols, include_category, scale=True, select_k=40)),
                    ("model", SVR(C=8.0, epsilon=0.25, gamma="scale")),
                ]
            ),
            include_category,
        ),
        ModelSpec(
            f"random_forest{'_cat' if include_category else ''}",
            Pipeline(
                [
                    ("prep", preprocess(numeric_cols, include_category, scale=False)),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=260,
                            min_samples_leaf=4,
                            max_features="sqrt",
                            random_state=7,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            include_category,
        ),
        ModelSpec(
            f"extra_trees{'_cat' if include_category else ''}",
            Pipeline(
                [
                    ("prep", preprocess(numeric_cols, include_category, scale=False)),
                    (
                        "model",
                        ExtraTreesRegressor(
                            n_estimators=320,
                            min_samples_leaf=3,
                            max_features="sqrt",
                            random_state=7,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            include_category,
        ),
        ModelSpec(
            f"gradient_boosting{'_cat' if include_category else ''}",
            Pipeline(
                [
                    ("prep", preprocess(numeric_cols, include_category, scale=False)),
                    (
                        "model",
                        GradientBoostingRegressor(
                            n_estimators=300,
                            learning_rate=0.025,
                            max_depth=3,
                            min_samples_leaf=6,
                            subsample=0.85,
                            random_state=7,
                        ),
                    ),
                ]
            ),
            include_category,
        ),
    ]
    return specs


def score_prediction(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(y) & np.isfinite(pred)
    return {
        "spearman_rho": float(stats.spearmanr(y[finite], pred[finite]).statistic),
        "pearson_r": float(stats.pearsonr(y[finite], pred[finite]).statistic),
        "mae_db": float(mean_absolute_error(y[finite], pred[finite])),
        "rmse_db": float(math.sqrt(mean_squared_error(y[finite], pred[finite]))),
    }


def oof_predictions(estimator: object, x: pd.DataFrame, y: np.ndarray, splitter, groups=None) -> np.ndarray:
    pred = np.full(len(y), np.nan, dtype=float)
    split_iter = splitter.split(x, y, groups) if groups is not None else splitter.split(x, y)
    for train_idx, test_idx in split_iter:
        est = clone(estimator)
        est.fit(x.iloc[train_idx], y[train_idx])
        pred[test_idx] = est.predict(x.iloc[test_idx])
    return pred


def evaluate_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df["nafnet_psnr"].to_numpy(dtype=float)
    groups = df["category"].to_numpy()
    all_predictions = []
    summaries = []
    modes = {
        "foggy_only": numeric_feature_columns(df, "foggy_only"),
        "paired_only": numeric_feature_columns(df, "paired_only"),
        "all_numeric": numeric_feature_columns(df, "all_numeric"),
    }
    random_cv = KFold(n_splits=5, shuffle=True, random_state=11)
    category_cv = LeaveOneGroupOut()

    for mode, numeric_cols in modes.items():
        for include_category in (False, True):
            if mode == "foggy_only" and include_category:
                continue
            x_cols = numeric_cols + (["category"] if include_category else [])
            x = df[x_cols].copy()
            for spec in model_specs(numeric_cols, include_category):
                print(f"evaluating {mode} / {spec.name}", flush=True)
                in_sample_est = clone(spec.estimator)
                in_sample_est.fit(x, y)
                in_pred = in_sample_est.predict(x)
                in_score = score_prediction(y, in_pred)
                summaries.append(
                    {
                        "feature_set": mode,
                        "model": spec.name,
                        "validation": "in_sample",
                        "n_features": len(numeric_cols),
                        **in_score,
                    }
                )
                all_predictions.append(
                    pd.DataFrame(
                        {
                            "category": df["category"],
                            "image_name": df["image_name"],
                            "nafnet_psnr": y,
                            "feature_set": mode,
                            "model": spec.name,
                            "validation": "in_sample",
                            "prediction": in_pred,
                        }
                    )
                )

                pred = oof_predictions(spec.estimator, x, y, random_cv)
                random_score = score_prediction(y, pred)
                summaries.append(
                    {
                        "feature_set": mode,
                        "model": spec.name,
                        "validation": "random_5fold_oof",
                        "n_features": len(numeric_cols),
                        **random_score,
                    }
                )
                all_predictions.append(
                    pd.DataFrame(
                        {
                            "category": df["category"],
                            "image_name": df["image_name"],
                            "nafnet_psnr": y,
                            "feature_set": mode,
                            "model": spec.name,
                            "validation": "random_5fold_oof",
                            "prediction": pred,
                        }
                    )
                )

                pred_logo = oof_predictions(spec.estimator, x, y, category_cv, groups=groups)
                logo_score = score_prediction(y, pred_logo)
                summaries.append(
                    {
                        "feature_set": mode,
                        "model": spec.name,
                        "validation": "leave_one_category_out",
                        "n_features": len(numeric_cols),
                        **logo_score,
                    }
                )
                all_predictions.append(
                    pd.DataFrame(
                        {
                            "category": df["category"],
                            "image_name": df["image_name"],
                            "nafnet_psnr": y,
                            "feature_set": mode,
                            "model": spec.name,
                            "validation": "leave_one_category_out",
                            "prediction": pred_logo,
                        }
                    )
                )

    summary = pd.DataFrame(summaries).sort_values(["validation", "spearman_rho"], ascending=[True, False])
    predictions = pd.concat(all_predictions, ignore_index=True)
    summary.to_csv(OUTDIR / "model_cv_summary.csv", index=False)
    predictions.to_csv(OUTDIR / "model_predictions.csv", index=False)
    with open(OUTDIR / "feature_sets.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in modes.items()}, f, indent=2)
    return summary, predictions


def plot_outputs(df: pd.DataFrame, corr: pd.DataFrame, model_summary: pd.DataFrame, predictions: pd.DataFrame) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    top = corr.head(18).copy().iloc[::-1]
    colors = ["#315c9b" if r >= 0 else "#b64545" for r in top["spearman_rho"]]
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    ax.barh(top["column"], top["spearman_rho"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Spearman correlation with NAFNet PSNR")
    ax.set_title("Best single metrics after deep paired-image search")
    ax.grid(axis="x", color="0.88", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "top_single_metric_correlations.png", dpi=240)
    plt.close(fig)

    best_col = str(corr.iloc[0]["column"])
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    for cat, sub in df.groupby("category"):
        ax.scatter(sub[best_col], sub["nafnet_psnr"], s=22, alpha=0.70, label=cat)
    ax.set_xlabel(best_col)
    ax.set_ylabel("NAFNet PSNR (dB)")
    ax.set_title(f"Best single metric: rho={corr.iloc[0]['spearman_rho']:.3f}")
    ax.grid(color="0.88", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGDIR / "best_single_metric_scatter.png", dpi=240)
    plt.close(fig)

    oof = model_summary[model_summary["validation"] == "random_5fold_oof"].copy()
    oof = oof.sort_values("spearman_rho", ascending=False).head(14).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.barh(oof["feature_set"] + " / " + oof["model"], oof["spearman_rho"], color="#4f8a57")
    ax.axvline(0.85, color="#b64545", linestyle="--", linewidth=1.2, label="goal 0.85")
    ax.set_xlabel("OOF Spearman correlation")
    ax.set_title("Composite predictors: random 5-fold validation")
    ax.grid(axis="x", color="0.88", linewidth=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "best_model_random_cv_rank.png", dpi=240)
    plt.close(fig)

    best = model_summary[model_summary["validation"] == "random_5fold_oof"].sort_values("spearman_rho", ascending=False).iloc[0]
    pred = predictions[
        (predictions["feature_set"] == best["feature_set"])
        & (predictions["model"] == best["model"])
        & (predictions["validation"] == "random_5fold_oof")
    ]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for cat, sub in pred.groupby("category"):
        ax.scatter(sub["prediction"], sub["nafnet_psnr"], s=22, alpha=0.72, label=cat)
    lo = min(pred["prediction"].min(), pred["nafnet_psnr"].min())
    hi = max(pred["prediction"].max(), pred["nafnet_psnr"].max())
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
    ax.set_xlabel("Out-of-fold predicted PSNR (dB)")
    ax.set_ylabel("Observed NAFNet PSNR (dB)")
    ax.set_title(f"Best OOF composite: rho={best['spearman_rho']:.3f}")
    ax.grid(color="0.88", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGDIR / "best_model_random_cv_scatter.png", dpi=240)
    plt.close(fig)


def write_summary(corr: pd.DataFrame, model_summary: pd.DataFrame) -> None:
    def markdown_table(frame: pd.DataFrame, floatfmt: str = ".4f") -> str:
        data = frame.copy()
        for col in data.columns:
            if pd.api.types.is_float_dtype(data[col]):
                data[col] = data[col].map(lambda x: format(float(x), floatfmt) if pd.notna(x) else "")
        rows = [list(data.columns)] + data.astype(str).values.tolist()
        widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]

        def fmt_row(row: list[str]) -> str:
            return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)) + " |"

        sep = "| " + " | ".join("-" * w for w in widths) + " |"
        return "\n".join([fmt_row(rows[0]), sep] + [fmt_row(row) for row in rows[1:]])

    top_single = corr.head(12)
    best_random = model_summary[model_summary["validation"] == "random_5fold_oof"].sort_values(
        "spearman_rho", ascending=False
    ).head(10)
    best_group = model_summary[model_summary["validation"] == "leave_one_category_out"].sort_values(
        "spearman_rho", ascending=False
    ).head(8)
    best_in = model_summary[model_summary["validation"] == "in_sample"].sort_values(
        "spearman_rho", ascending=False
    ).head(8)
    lines = [
        "# Deep metric-to-PSNR search",
        "",
        "Target: find paired or unpaired image metrics that correlate with NAFNet PSNR on the 552-image held-out fog set.",
        "",
        "## Best single metrics",
        "",
        markdown_table(top_single[
            ["column", "spearman_rho", "pearson_r", "category_centered_spearman_rho"]
        ]),
        "",
        "## Best composite predictors, random 5-fold out-of-fold",
        "",
        markdown_table(best_random[
            ["feature_set", "model", "spearman_rho", "pearson_r", "mae_db", "rmse_db"]
        ]),
        "",
        "## Best composite predictors, leave-one-category-out",
        "",
        markdown_table(best_group[
            ["feature_set", "model", "spearman_rho", "pearson_r", "mae_db", "rmse_db"]
        ]),
        "",
        "## Highest in-sample composite scores",
        "",
        markdown_table(best_in[
            ["feature_set", "model", "spearman_rho", "pearson_r", "mae_db", "rmse_db"]
        ]),
        "",
        "Interpretation note: in-sample composite scores are an overfit upper bound. The random 5-fold out-of-fold score is the most relevant estimate for images drawn from the same six-category test distribution; leave-one-category-out is a harder test of whether the metric generalizes to unseen content classes.",
        "",
    ]
    (OUTDIR / "deep_dive_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(BASE_TABLE)
    blur = pd.read_csv(BLUR_TABLE)
    rich = compute_rich_metrics(base)
    df = merge_features(base, blur, rich)
    df.to_csv(OUTDIR / "all_metric_features.csv", index=False)
    corr = single_metric_correlations(df)
    model_summary, predictions = evaluate_models(df)
    plot_outputs(df, corr, model_summary, predictions)
    write_summary(corr, model_summary)
    print(f"wrote outputs to {OUTDIR}")


if __name__ == "__main__":
    main()
