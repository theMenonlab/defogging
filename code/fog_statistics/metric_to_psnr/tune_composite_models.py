#!/usr/bin/env python3
"""Focused tuning pass for composite fog metrics.

This reads the cached feature table from deep_dive_metric_search.py and tries a
small set of stronger regressors/ensembles. The goal is to see whether the
validated Spearman correlation can approach 0.85 without recomputing image
features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import SVR

from deep_dive_metric_search import OUTDIR, numeric_feature_columns, oof_predictions, preprocess, score_prediction


DF_PATH = OUTDIR / "all_metric_features.csv"
TUNE_SUMMARY = OUTDIR / "model_tuning_summary.csv"
TUNE_PREDS = OUTDIR / "model_tuning_predictions.csv"


def make_pipeline(numeric_cols: list[str], estimator, *, scale: bool = False, select_k: int | None = None, quantile: bool = False) -> Pipeline:
    steps = [("prep", preprocess(numeric_cols, include_category=False, scale=scale, select_k=select_k))]
    if quantile:
        steps.append(("quantile", QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=23)))
    steps.append(("model", estimator))
    return Pipeline(steps)


def candidate_models(numeric_cols: list[str]) -> list[tuple[str, Pipeline]]:
    models: list[tuple[str, Pipeline]] = []
    for depth, lr, leaf, subsample, n_est in [
        (2, 0.035, 4, 0.90, 420),
        (2, 0.025, 5, 0.85, 650),
        (3, 0.025, 4, 0.85, 500),
        (3, 0.018, 5, 0.85, 750),
        (3, 0.035, 8, 0.80, 420),
        (4, 0.018, 8, 0.85, 600),
    ]:
        name = f"gbr_d{depth}_lr{lr:g}_leaf{leaf}_sub{subsample:g}_n{n_est}"
        models.append(
            (
                name,
                make_pipeline(
                    numeric_cols,
                    GradientBoostingRegressor(
                        n_estimators=n_est,
                        learning_rate=lr,
                        max_depth=depth,
                        min_samples_leaf=leaf,
                        subsample=subsample,
                        random_state=31,
                    ),
                ),
            )
        )
    for leaves, lr, l2, iters in [
        (15, 0.035, 0.0, 520),
        (23, 0.030, 0.01, 620),
        (31, 0.025, 0.03, 720),
        (47, 0.020, 0.05, 850),
    ]:
        name = f"hgb_leaf{leaves}_lr{lr:g}_l2{l2:g}_n{iters}"
        models.append(
            (
                name,
                make_pipeline(
                    numeric_cols,
                    HistGradientBoostingRegressor(
                        max_iter=iters,
                        learning_rate=lr,
                        max_leaf_nodes=leaves,
                        l2_regularization=l2,
                        min_samples_leaf=8,
                        random_state=31,
                    ),
                ),
            )
        )
    for leaf, max_features in [(1, "sqrt"), (2, "sqrt"), (3, "sqrt"), (3, 0.6), (5, "sqrt")]:
        models.append(
            (
                f"extra_trees_leaf{leaf}_mf{max_features}",
                make_pipeline(
                    numeric_cols,
                    ExtraTreesRegressor(
                        n_estimators=700,
                        min_samples_leaf=leaf,
                        max_features=max_features,
                        bootstrap=False,
                        random_state=31,
                        n_jobs=-1,
                    ),
                ),
            )
        )
    for leaf, max_features in [(2, "sqrt"), (3, "sqrt"), (4, 0.6)]:
        models.append(
            (
                f"random_forest_leaf{leaf}_mf{max_features}",
                make_pipeline(
                    numeric_cols,
                    RandomForestRegressor(
                        n_estimators=550,
                        min_samples_leaf=leaf,
                        max_features=max_features,
                        random_state=31,
                        n_jobs=-1,
                    ),
                ),
            )
        )
    models.extend(
        [
            (
                "svr_select80_c12",
                make_pipeline(numeric_cols, SVR(C=12.0, epsilon=0.18, gamma="scale"), scale=True, select_k=80),
            ),
            (
                "svr_select120_c16_quantile",
                make_pipeline(numeric_cols, SVR(C=16.0, epsilon=0.15, gamma="scale"), scale=True, select_k=120, quantile=True),
            ),
            (
                "kernel_ridge_rbf_select80",
                make_pipeline(numeric_cols, KernelRidge(alpha=0.8, kernel="rbf", gamma=0.01), scale=True, select_k=80),
            ),
        ]
    )
    return models


def evaluate_one(name: str, estimator: Pipeline, x: pd.DataFrame, y: np.ndarray, cv, groups=None) -> tuple[dict[str, float | str], np.ndarray]:
    pred = oof_predictions(estimator, x, y, cv, groups=groups)
    score = score_prediction(y, pred)
    return {"model": name, **score}, pred


def rank_average(preds: list[np.ndarray]) -> np.ndarray:
    ranks = [stats.rankdata(p, method="average") for p in preds]
    return np.mean(ranks, axis=0)


def main() -> None:
    df = pd.read_csv(DF_PATH)
    y = df["nafnet_psnr"].to_numpy(dtype=float)
    groups = df["category"].to_numpy()
    random_cv = KFold(n_splits=5, shuffle=True, random_state=11)
    logo_cv = LeaveOneGroupOut()
    all_rows = []
    pred_rows = []

    for feature_set in ("paired_only", "all_numeric"):
        numeric_cols = numeric_feature_columns(df, feature_set)
        x = df[numeric_cols].copy()
        random_preds: dict[str, np.ndarray] = {}
        models = candidate_models(numeric_cols)
        for name, estimator in models:
            print(f"random CV {feature_set} / {name}", flush=True)
            row, pred = evaluate_one(name, estimator, x, y, random_cv)
            row.update({"feature_set": feature_set, "validation": "random_5fold_oof", "n_features": len(numeric_cols)})
            all_rows.append(row)
            random_preds[name] = pred
            pred_rows.append(
                pd.DataFrame(
                    {
                        "category": df["category"],
                        "image_name": df["image_name"],
                        "feature_set": feature_set,
                        "model": name,
                        "validation": "random_5fold_oof",
                        "nafnet_psnr": y,
                        "prediction": pred,
                    }
                )
            )

        ranked = sorted(all_rows, key=lambda r: (r["feature_set"] == feature_set, r["spearman_rho"]), reverse=True)
        top_names = [r["model"] for r in ranked if r["feature_set"] == feature_set and r["validation"] == "random_5fold_oof"][:6]
        for k in (2, 3, 5, 6):
            names = top_names[:k]
            pred = rank_average([random_preds[n] for n in names])
            row = {"model": f"rank_average_top{k}", **score_prediction(y, pred)}
            row.update({"feature_set": feature_set, "validation": "random_5fold_oof", "n_features": len(numeric_cols)})
            all_rows.append(row)
            pred_rows.append(
                pd.DataFrame(
                    {
                        "category": df["category"],
                        "image_name": df["image_name"],
                        "feature_set": feature_set,
                        "model": f"rank_average_top{k}",
                        "validation": "random_5fold_oof",
                        "nafnet_psnr": y,
                        "prediction": pred,
                    }
                )
            )

        for name in top_names[:5]:
            estimator = dict(models)[name]
            print(f"leave-one-category {feature_set} / {name}", flush=True)
            row, pred = evaluate_one(name, estimator, x, y, logo_cv, groups=groups)
            row.update({"feature_set": feature_set, "validation": "leave_one_category_out", "n_features": len(numeric_cols)})
            all_rows.append(row)
            pred_rows.append(
                pd.DataFrame(
                    {
                        "category": df["category"],
                        "image_name": df["image_name"],
                        "feature_set": feature_set,
                        "model": name,
                        "validation": "leave_one_category_out",
                        "nafnet_psnr": y,
                        "prediction": pred,
                    }
                )
            )

    summary = pd.DataFrame(all_rows).sort_values(["validation", "spearman_rho"], ascending=[True, False])
    predictions = pd.concat(pred_rows, ignore_index=True)
    summary.to_csv(TUNE_SUMMARY, index=False)
    predictions.to_csv(TUNE_PREDS, index=False)
    print(summary.groupby("validation").head(12).to_string(index=False))


if __name__ == "__main__":
    main()
