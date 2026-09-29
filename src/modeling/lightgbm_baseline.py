#!/usr/bin/env python3
"""LightGBM baseline for the LCT collector risk task.

The baseline follows the EDA interpretation:
- entity: sensor channel;
- time step: one day;
- target: whether the channel will have at least one alarm in the next 24 hours;
- validation: chronological train/validation/test split on 2025-2026 data.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import ks_2samp
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
REPORT = ROOT / "reports" / "baseline_lightgbm"
TABLES = REPORT / "tables"
FIGURES = REPORT / "figures"
MODELS = REPORT / "models"

try:
    from .daily_data import TEXT_FLAGS, load_dictionaries, combine_daily, read_year_daily
except ImportError:
    from daily_data import TEXT_FLAGS, load_dictionaries, combine_daily, read_year_daily


@dataclass(frozen=True)
class SplitBounds:
    train_end: str = "2025-09-30"
    valid_end: str = "2025-12-31"


def ensure_dirs() -> None:
    for path in [REPORT, TABLES, FIGURES, MODELS]:
        path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and evaluate the LightGBM baseline.")
    parser.add_argument("--years", nargs="+", default=["2025", "2026"], help="Journal years to use.")
    parser.add_argument("--chunksize", type=int, default=1_000_000, help="Rows per raw CSV chunk.")
    parser.add_argument("--max-train-rows", type=int, default=700_000, help="Training sample cap.")
    parser.add_argument("--max-eval-rows", type=int, default=1_200_000, help="Validation/test scoring cap.")
    parser.add_argument("--shap-rows", type=int, default=4_000, help="Rows for SHAP importance.")
    parser.add_argument("--perm-rows", type=int, default=5_000, help="Rows for permutation importance.")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def build_daily_mart(years: list[str], chunksize: int) -> pd.DataFrame:
    meta = load_dictionaries()
    daily_parts = [read_year_daily(year, chunksize) for year in years]
    daily = combine_daily(daily_parts)

    channels = pd.Index(sorted(set(daily["channel_id"]) | set(meta["channel_id"])), name="channel_id")
    dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D", name="date")
    grid = pd.MultiIndex.from_product([channels, dates]).to_frame(index=False)

    mart = grid.merge(daily, on=["channel_id", "date"], how="left")
    count_cols = [
        "events_count",
        "alarm_count",
        "numeric_count",
        "text_count",
        "missing_count",
        "negative_count",
        "sentinel_count",
        "value_sum",
        "value_sumsq",
        *TEXT_FLAGS.keys(),
    ]
    for col in count_cols:
        mart[col] = mart[col].fillna(0)
    mart["value_min"] = mart["value_min"].astype("float32")
    mart["value_max"] = mart["value_max"].astype("float32")

    mart = mart.merge(meta, on="channel_id", how="left")
    for col in ["engineering_system", "sensor_type", "object_id", "object_kind"]:
        mart[col] = mart[col].fillna("Нет в справочнике").astype("category")

    mart["year"] = mart["date"].dt.year.astype("int16")
    mart["month"] = mart["date"].dt.month.astype("int8")
    mart["dayofweek"] = mart["date"].dt.dayofweek.astype("int8")
    mart["is_weekend"] = mart["dayofweek"].isin([5, 6]).astype("int8")

    safe_events = mart["events_count"].replace(0, np.nan)
    safe_numeric = mart["numeric_count"].replace(0, np.nan)
    mart["alarm_rate_1d"] = (mart["alarm_count"] / safe_events).fillna(0).astype("float32")
    mart["numeric_share_1d"] = (mart["numeric_count"] / safe_events).fillna(0).astype("float32")
    mart["text_share_1d"] = (mart["text_count"] / safe_events).fillna(0).astype("float32")
    mart["missing_share_1d"] = (mart["missing_count"] / safe_events).fillna(0).astype("float32")
    mart["value_mean_1d"] = (mart["value_sum"] / safe_numeric).astype("float32")
    variance = mart["value_sumsq"] / safe_numeric - (mart["value_mean_1d"].astype("float64") ** 2)
    mart["value_std_1d"] = np.sqrt(np.maximum(variance, 0)).astype("float32")
    mart["value_range_1d"] = (mart["value_max"] - mart["value_min"]).astype("float32")

    mart = mart.sort_values(["channel_id", "date"]).reset_index(drop=True)
    rolling_cols = [
        "events_count",
        "alarm_count",
        "numeric_count",
        "text_count",
        "missing_count",
        "negative_count",
        "sentinel_count",
        "fault_count",
        "no_power_count",
        "undefined_count",
        "movement_detected_count",
        "open_count",
    ]
    for window in [3, 7, 14]:
        rolled = (
            mart.groupby("channel_id", observed=True)[rolling_cols]
            .rolling(window=window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        for col in rolling_cols:
            mart[f"{col}_sum_{window}d"] = rolled[col].astype("float32")
        mart[f"alarm_rate_{window}d"] = (
            mart[f"alarm_count_sum_{window}d"] / mart[f"events_count_sum_{window}d"].replace(0, np.nan)
        ).fillna(0).astype("float32")
        mart[f"numeric_share_{window}d"] = (
            mart[f"numeric_count_sum_{window}d"] / mart[f"events_count_sum_{window}d"].replace(0, np.nan)
        ).fillna(0).astype("float32")

    mart["target_alarm_next_24h"] = (
        mart.groupby("channel_id", observed=True)["alarm_count"].shift(-1).fillna(0).gt(0).astype("int8")
    )
    mart = mart[mart["date"] < mart["date"].max()].copy()
    return mart


def add_split(df: pd.DataFrame, bounds: SplitBounds) -> pd.DataFrame:
    train_end = pd.Timestamp(bounds.train_end)
    valid_end = pd.Timestamp(bounds.valid_end)
    df["split"] = np.select(
        [df["date"] <= train_end, df["date"] <= valid_end],
        ["train", "valid"],
        default="test",
    )
    return df


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    ignore = {
        "date",
        "channel_id",
        "target_alarm_next_24h",
        "split",
        "value_sum",
        "value_sumsq",
    }
    features = [col for col in df.columns if col not in ignore]
    categorical = ["engineering_system", "sensor_type", "object_id", "object_kind"]
    return features, categorical


def sample_training(df: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.copy()
    pos = df[df["target_alarm_next_24h"] == 1]
    neg = df[df["target_alarm_next_24h"] == 0]
    pos_cap = min(len(pos), max(1, int(max_rows * 0.45)))
    neg_cap = max_rows - pos_cap
    pos_sample = pos.sample(n=pos_cap, random_state=random_state) if len(pos) > pos_cap else pos
    neg_sample = neg.sample(n=min(len(neg), neg_cap), random_state=random_state)
    return pd.concat([pos_sample, neg_sample], ignore_index=True).sample(frac=1, random_state=random_state)


def sample_eval(df: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.copy()
    return df.sample(n=max_rows, random_state=random_state)


def train_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    features: list[str],
    categorical: list[str],
    random_state: int,
) -> lgb.LGBMClassifier:
    y_train = train_df["target_alarm_next_24h"]
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / max(pos, 1)
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=700,
        learning_rate=0.035,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=120,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.2,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(
        train_df[features],
        y_train,
        eval_set=[(valid_df[features], valid_df["target_alarm_next_24h"])],
        eval_metric=["auc", "average_precision"],
        categorical_feature=categorical,
        callbacks=[lgb.early_stopping(60), lgb.log_evaluation(50)],
    )
    return model


def predict_split(model: lgb.LGBMClassifier, df: pd.DataFrame, features: list[str], max_rows: int, random_state: int) -> pd.DataFrame:
    eval_df = sample_eval(df, max_rows=max_rows, random_state=random_state)
    out = eval_df[["date", "channel_id", "split", "target_alarm_next_24h"]].copy()
    out["score"] = model.predict_proba(eval_df[features])[:, 1]
    return out


def best_threshold(y_true: np.ndarray, score: np.ndarray) -> tuple[float, pd.DataFrame]:
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    rows = []
    best_thr = 0.5
    best_f1 = -1.0
    for p, r, thr in zip(precision[:-1], recall[:-1], thresholds):
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        rows.append({"threshold": float(thr), "precision": float(p), "recall": float(r), "f1": float(f1)})
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr, pd.DataFrame(rows)


def metric_row(name: str, y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float | str | int]:
    pred = (score >= threshold).astype(int)
    return {
        "split": name,
        "rows": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) > 1 else np.nan,
        "pr_auc": float(average_precision_score(y_true, score)) if len(np.unique(y_true)) > 1 else np.nan,
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def save_curves(preds: pd.DataFrame) -> None:
    colors = {"train": "#2563eb", "valid": "#16a34a", "test": "#dc2626"}

    plt.figure(figsize=(8, 6))
    for split, part in preds.groupby("split"):
        y = part["target_alarm_next_24h"].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, part["score"].to_numpy())
        auc = roc_auc_score(y, part["score"].to_numpy())
        plt.plot(fpr, tpr, label=f"{split}: ROC-AUC={auc:.3f}", color=colors.get(split))
    plt.plot([0, 1], [0, 1], "--", color="#9ca3af")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривая LightGBM baseline")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES / "roc_auc_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 6))
    for split, part in preds.groupby("split"):
        y = part["target_alarm_next_24h"].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y, part["score"].to_numpy())
        ap = average_precision_score(y, part["score"].to_numpy())
        plt.plot(recall, precision, label=f"{split}: PR-AUC={ap:.3f}", color=colors.get(split))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR-кривая LightGBM baseline")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES / "pr_curve.png", dpi=180)
    plt.close()


def save_target_plots(df: pd.DataFrame, preds: pd.DataFrame) -> None:
    dist = (
        df.groupby(["split", "target_alarm_next_24h"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    dist["target"] = dist["target_alarm_next_24h"].map({0: "target=0", 1: "target=1"})
    dist.to_csv(TABLES / "target_distribution.csv", index=False)

    pivot = dist.pivot(index="split", columns="target", values="rows").fillna(0)
    pivot = pivot.reindex(["train", "valid", "test"])
    ax = pivot.plot(kind="bar", stacked=False, figsize=(9, 5), color=["#94a3b8", "#ef4444"])
    ax.set_title("Распределение target=0 и target=1 по выборкам")
    ax.set_xlabel("Выборка")
    ax.set_ylabel("Число строк витрины")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES / "target_distribution_by_split.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, 1, 41)
    for split, part in preds.groupby("split"):
        for target, linestyle in [(0, "-"), (1, "--")]:
            scores = part.loc[part["target_alarm_next_24h"] == target, "score"]
            if scores.empty:
                continue
            hist, edges = np.histogram(scores, bins=bins, density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            plt.plot(centers, hist, linestyle=linestyle, label=f"{split}, target={target}")
    plt.xlabel("Скор LightGBM")
    plt.ylabel("Плотность")
    plt.title("Распределение скоринга для target=0 и target=1")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES / "score_distribution_by_target.png", dpi=180)
    plt.close()


def save_threshold_plot(threshold_df: pd.DataFrame) -> None:
    threshold_df.to_csv(TABLES / "threshold_metrics_valid.csv", index=False)
    slim = threshold_df.sort_values("threshold").iloc[:: max(1, len(threshold_df) // 300)]
    plt.figure(figsize=(9, 6))
    plt.plot(slim["threshold"], slim["precision"], label="Precision", color="#2563eb")
    plt.plot(slim["threshold"], slim["recall"], label="Recall", color="#dc2626")
    plt.plot(slim["threshold"], slim["f1"], label="F1", color="#16a34a")
    plt.xlabel("Порог")
    plt.ylabel("Метрика")
    plt.title("Precision, Recall, F1 в зависимости от порога")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES / "threshold_precision_recall_f1.png", dpi=180)
    plt.close()


def normalize_importance(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.maximum(values, 0)
    total = values.sum()
    if total == 0:
        return np.zeros_like(values)
    return values / total


def plot_importance(table: pd.DataFrame, value_col: str, filename: str, title: str, top_n: int = 25) -> None:
    top = table.sort_values(value_col, ascending=False).head(top_n).sort_values(value_col)
    plt.figure(figsize=(9, max(5, 0.32 * len(top))))
    plt.barh(top["feature"], top[value_col], color="#2563eb")
    plt.xlabel(value_col)
    plt.title(title)
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=180)
    plt.close()


def save_importances(
    model: lgb.LGBMClassifier,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    random_state: int,
    shap_rows: int,
    perm_rows: int,
) -> pd.DataFrame:
    booster = model.booster_
    split_imp = booster.feature_importance(importance_type="split")
    gain_imp = booster.feature_importance(importance_type="gain")
    base = pd.DataFrame({"feature": features, "split": split_imp, "gain": gain_imp})

    shap_sample = sample_eval(test_df, shap_rows, random_state)
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(shap_sample[features])
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_imp = np.abs(np.asarray(shap_values)).mean(axis=0)
    base["shap"] = shap_imp

    perm_sample = sample_eval(test_df, perm_rows, random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perm = permutation_importance(
            model,
            perm_sample[features],
            perm_sample["target_alarm_next_24h"],
            scoring="average_precision",
            n_repeats=5,
            random_state=random_state,
            n_jobs=-1,
        )
    base["permutation"] = np.maximum(perm.importances_mean, 0)

    base["split_norm"] = normalize_importance(base["split"].to_numpy())
    base["gain_norm"] = normalize_importance(base["gain"].to_numpy())
    base["shap_norm"] = normalize_importance(base["shap"].to_numpy())
    base["permutation_norm"] = normalize_importance(base["permutation"].to_numpy())
    base["total_normalized_importance"] = (
        base["split_norm"] + base["gain_norm"] + base["shap_norm"] + base["permutation_norm"]
    )

    for col in ["split", "gain", "shap", "permutation"]:
        out = base[["feature", col, f"{col}_norm"]].sort_values(col, ascending=False)
        out.to_csv(TABLES / f"feature_importance_{col}.csv", index=False)
        plot_importance(out.rename(columns={col: "importance"}), "importance", f"feature_importance_{col}.png", f"Важность признаков: {col}")
        if col == "gain":
            out.to_csv(TABLES / "feature_importance_grain_gain_alias.csv", index=False)
            plot_importance(
                out.rename(columns={col: "importance"}),
                "importance",
                "feature_importance_grain_gain_alias.png",
                "Важность признаков: gain (grain в запросе)",
            )

    base = base.sort_values("total_normalized_importance", ascending=False)
    base.to_csv(TABLES / "feature_importance_combined.csv", index=False)
    base.to_parquet(TABLES / "feature_importance_combined.parquet", index=False)
    plot_importance(
        base,
        "total_normalized_importance",
        "feature_importance_total_normalized.png",
        "Суммарная нормированная важность признаков",
    )
    return base


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = expected.dropna()
    actual = actual.dropna()
    if expected.nunique() < 2 or actual.empty:
        return np.nan
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.nanquantile(expected, quantiles))
    if len(edges) < 3:
        return np.nan
    edges[0] = -np.inf
    edges[-1] = np.inf
    exp_counts = pd.cut(expected, edges, include_lowest=True).value_counts(sort=False)
    act_counts = pd.cut(actual, edges, include_lowest=True).value_counts(sort=False)
    exp_pct = exp_counts / max(exp_counts.sum(), 1)
    act_pct = act_counts / max(act_counts.sum(), 1)
    eps = 1e-6
    return float(((act_pct + eps) - (exp_pct + eps)).mul(np.log((act_pct + eps) / (exp_pct + eps))).sum())


def save_drift_and_coverage(df: pd.DataFrame, features: list[str], categorical: list[str], random_state: int) -> pd.DataFrame:
    coverage_rows = []
    for split, part in df.groupby("split", observed=True):
        for feature in features:
            s = part[feature]
            coverage_rows.append(
                {
                    "split": split,
                    "feature": feature,
                    "dtype": "categorical" if feature in categorical else "numeric",
                    "rows": int(len(s)),
                    "non_null": int(s.notna().sum()),
                    "coverage": float(s.notna().mean()),
                    "zero_share": float((s.fillna(0) == 0).mean()) if feature not in categorical else np.nan,
                    "unique_values": int(s.nunique(dropna=True)),
                }
            )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_parquet(TABLES / "feature_coverage.parquet", index=False)
    coverage.to_csv(TABLES / "feature_coverage.csv", index=False)

    numeric_features = [f for f in features if f not in categorical]
    split_samples = {
        split: sample_eval(part, 120_000, random_state)
        for split, part in df.groupby("split", observed=True)
    }
    train = split_samples["train"]
    drift_rows = []
    for feature in numeric_features:
        train_s = train[feature]
        row = {
            "feature": feature,
            "train_mean": float(train_s.mean()),
            "train_std": float(train_s.std()),
        }
        for split in ["valid", "test"]:
            part_s = split_samples[split][feature]
            row[f"{split}_mean"] = float(part_s.mean())
            row[f"{split}_std"] = float(part_s.std())
            row[f"psi_train_{split}"] = psi(train_s, part_s)
            if train_s.nunique(dropna=True) > 1 and part_s.nunique(dropna=True) > 1:
                row[f"ks_train_{split}"] = float(ks_2samp(train_s.dropna(), part_s.dropna()).statistic)
            else:
                row[f"ks_train_{split}"] = np.nan
        drift_rows.append(row)

    drift = pd.DataFrame(drift_rows)
    drift.to_parquet(TABLES / "feature_drift.parquet", index=False)
    drift.to_csv(TABLES / "feature_drift.csv", index=False)

    plot_drift_bars(drift, "psi_train_test", "drift_psi_train_test.png", "PSI признаков: train vs test")
    plot_drift_bars(drift, "ks_train_test", "drift_ks_train_test.png", "KS признаков: train vs test")
    plot_coverage(coverage)
    return drift


def plot_drift_bars(drift: pd.DataFrame, col: str, filename: str, title: str, top_n: int = 25) -> None:
    top = drift[["feature", col]].dropna().sort_values(col, ascending=False).head(top_n).sort_values(col)
    plt.figure(figsize=(9, max(5, 0.32 * len(top))))
    plt.barh(top["feature"], top[col], color="#f97316")
    plt.xlabel(col)
    plt.title(title)
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=180)
    plt.close()


def plot_coverage(coverage: pd.DataFrame) -> None:
    pivot = coverage.pivot(index="feature", columns="split", values="coverage")
    pivot["min_coverage"] = pivot.min(axis=1)
    pivot = pivot.sort_values("min_coverage").head(35).drop(columns="min_coverage")
    plt.figure(figsize=(7, max(6, 0.25 * len(pivot))))
    data = pivot[["train", "valid", "test"]].to_numpy()
    plt.imshow(data, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    plt.yticks(range(len(pivot)), pivot.index)
    plt.xticks(range(3), ["train", "valid", "test"])
    plt.colorbar(label="coverage")
    plt.title("Покрытие признаков по выборкам")
    plt.tight_layout()
    plt.savefig(FIGURES / "feature_coverage_heatmap.png", dpi=180)
    plt.close()


def df_to_markdown(df: pd.DataFrame) -> str:
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
        else:
            view[col] = view[col].astype(str)
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in view.to_numpy(dtype=str)]
    return "\n".join([header, sep, *rows])


def write_report(
    metrics: pd.DataFrame,
    best_thr: float,
    importances: pd.DataFrame,
    drift: pd.DataFrame,
    rows_total: int,
    train_rows: int,
) -> None:
    top_importance = importances.head(15)[["feature", "total_normalized_importance"]]
    top_drift = drift[["feature", "psi_train_test", "ks_train_test"]].sort_values("psi_train_test", ascending=False).head(15)
    lines = [
        "# LightGBM baseline",
        "",
        "Baseline построен по выводам EDA: дневная витрина по каналу и target `target_alarm_next_24h`,",
        "то есть наличие хотя бы одного тревожного события по этому каналу в следующие 24 часа.",
        "",
        "## Постановка",
        "",
        "- Период: 2025-2026 годы.",
        "- Train: до 2025-09-30 включительно.",
        "- Validation: 2025-10-01 - 2025-12-31.",
        "- Test: с 2026-01-01 до последней доступной даты минус один день.",
        "- Сущность: `ид_канала_данных`.",
        "- Шаг времени: 1 день.",
        f"- Всего строк витрины: {rows_total:,}.",
        f"- Строк в обучающей подвыборке LightGBM: {train_rows:,}.",
        f"- Лучший порог по F1 на validation: {best_thr:.4f}.",
        "",
        "## Метрики",
        "",
        df_to_markdown(metrics),
        "",
        "## Топ суммарной нормированной важности",
        "",
        df_to_markdown(top_importance),
        "",
        "## Максимальный drift train vs test",
        "",
        df_to_markdown(top_drift),
        "",
        "## Артефакты",
        "",
        "- `figures/roc_auc_curve.png` - ROC-AUC.",
        "- `figures/pr_curve.png` - PR-кривая.",
        "- `figures/target_distribution_by_split.png` - распределение target=0/1.",
        "- `figures/score_distribution_by_target.png` - распределение скоринга по target.",
        "- `figures/threshold_precision_recall_f1.png` - Precision, Recall, F1 по порогам.",
        "- `figures/feature_importance_split.png` - важность LightGBM split.",
        "- `figures/feature_importance_gain.png` - важность LightGBM gain; это стандартная метрика, соответствующая `grain` из запроса.",
        "- `figures/feature_importance_grain_gain_alias.png` - тот же gain-график с alias-именем.",
        "- `figures/feature_importance_shap.png` - SHAP-важность.",
        "- `figures/feature_importance_permutation.png` - permutation importance.",
        "- `figures/feature_importance_total_normalized.png` - суммарная нормированная важность.",
        "- `figures/drift_psi_train_test.png` - PSI train vs test.",
        "- `figures/drift_ks_train_test.png` - KS train vs test.",
        "- `figures/feature_coverage_heatmap.png` - покрытие признаков.",
        "- `tables/feature_drift.parquet` - PSI/KS drift по признакам.",
        "- `tables/feature_coverage.parquet` - coverage признаков.",
        "",
        "## Ограничения",
        "",
        "Target является proxy-разметкой по техническому признаку `тревожное`, а не подтвержденной аварией.",
        "Baseline нужен как первая воспроизводимая точка отсчета, а не финальная бизнес-модель.",
    ]
    (REPORT / "baseline_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dirs()
    mart = build_daily_mart(args.years, args.chunksize)
    mart = add_split(mart, SplitBounds())
    features, categorical = feature_columns(mart)

    for col in categorical:
        mart[col] = mart[col].astype("category")

    train_full = mart[mart["split"] == "train"]
    valid_full = mart[mart["split"] == "valid"]
    test_full = mart[mart["split"] == "test"]
    train_df = sample_training(train_full, args.max_train_rows, args.random_state)
    valid_df = sample_eval(valid_full, args.max_eval_rows, args.random_state)
    test_df = sample_eval(test_full, args.max_eval_rows, args.random_state)

    print(f"[baseline] mart rows={len(mart):,}; train sample={len(train_df):,}; valid={len(valid_df):,}; test={len(test_df):,}")
    model = train_model(train_df, valid_df, features, categorical, args.random_state)
    model.booster_.save_model(str(MODELS / "lightgbm_baseline.txt"))

    preds = pd.concat(
        [
            predict_split(model, train_full, features, args.max_eval_rows, args.random_state),
            predict_split(model, valid_full, features, args.max_eval_rows, args.random_state),
            predict_split(model, test_full, features, args.max_eval_rows, args.random_state),
        ],
        ignore_index=True,
    )
    preds.to_parquet(TABLES / "predictions_sample.parquet", index=False)

    valid_pred = preds[preds["split"] == "valid"]
    best_thr, threshold_df = best_threshold(
        valid_pred["target_alarm_next_24h"].to_numpy(),
        valid_pred["score"].to_numpy(),
    )
    metric_rows = []
    for split, part in preds.groupby("split", observed=True):
        metric_rows.append(
            metric_row(
                split,
                part["target_alarm_next_24h"].to_numpy(),
                part["score"].to_numpy(),
                best_thr,
            )
        )
    metrics = pd.DataFrame(metric_rows).sort_values("split")
    metrics.to_csv(TABLES / "metrics.csv", index=False)
    metrics.to_parquet(TABLES / "metrics.parquet", index=False)

    save_curves(preds)
    save_target_plots(mart, preds)
    save_threshold_plot(threshold_df)
    importances = save_importances(model, train_df, test_df, features, args.random_state, args.shap_rows, args.perm_rows)
    drift = save_drift_and_coverage(mart, features, categorical, args.random_state)

    metadata = {
        "years": args.years,
        "rows_total": int(len(mart)),
        "train_full_rows": int(len(train_full)),
        "valid_full_rows": int(len(valid_full)),
        "test_full_rows": int(len(test_full)),
        "train_sample_rows": int(len(train_df)),
        "features": features,
        "categorical_features": categorical,
        "best_threshold_valid_f1": best_thr,
    }
    (TABLES / "baseline_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(metrics, best_thr, importances, drift, len(mart), len(train_df))
    print(f"[baseline] done: {REPORT}")


if __name__ == "__main__":
    main()
