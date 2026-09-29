#!/usr/bin/env python3
"""Reproducible chronological comparison on the unchanged next-day alarm target.

Prediction time is END of feature day. This is not guaranteed 24h advance notice.
Select model/threshold on validation only; evaluate the selected model once on test.
Raw, cached matrices, models and row-level predictions stay outside git.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".mplconfig"))
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

import lightgbm_baseline as legacy

ROOT = legacy.ROOT
CACHE = ROOT / "data/interim"
OUT = ROOT / "reports/improved_v1"
TARGET = "target_alarm_next_24h"


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def fingerprint():
    inputs = [ROOT / "src/modeling/lightgbm_baseline.py", legacy.RAW / "import_manifest.json",
              ROOT / "src/modeling/daily_data.py"]
    return hashlib.sha256(b"".join(p.read_bytes() for p in inputs)).hexdigest()


def prepare():
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "baseline_mart_2025_2026.parquet"
    signature = CACHE / "baseline_mart_signature.txt"
    stamp = fingerprint()
    if path.exists() and signature.exists() and signature.read_text() == stamp:
        log("Read cached baseline matrix")
        return pd.read_parquet(path)
    original_reader = legacy.read_year_daily

    def cached_reader(year, chunksize):
        daily_path = CACHE / f"daily_{year}.parquet"
        daily_signature = daily_path.with_suffix(".signature")
        if daily_path.exists() and daily_signature.exists() and daily_signature.read_text() == stamp:
            return pd.read_parquet(daily_path)
        pickle_path = daily_path.with_suffix(".pkl")
        pickle_signature = pickle_path.with_suffix(".pkl.signature")
        if pickle_path.exists() and pickle_signature.exists() and pickle_signature.read_text() == stamp:
            result = pd.read_pickle(pickle_path)
        else:
            result = original_reader(year, chunksize)
        result.to_parquet(daily_path, index=False)
        daily_signature.write_text(stamp)
        log(f"Cached daily aggregates {year}: {len(result):,} rows")
        return result

    legacy.read_year_daily = cached_reader
    try:
        mart = legacy.build_daily_mart(["2025", "2026"], 500_000)
    finally:
        legacy.read_year_daily = original_reader
    mart = legacy.add_split(mart, legacy.SplitBounds())
    # Preserve original positional order for identical deterministic evaluation samples.
    mart.to_parquet(path, index=False)
    signature.write_text(stamp)
    log(f"Cached baseline matrix: {len(mart):,} rows")
    return mart


def rolling_sum(values, window):
    """Trailing inclusive sum per channel; never crosses channel boundaries."""
    totals = np.cumsum(values, axis=1, dtype=np.float64)
    result = totals.copy()
    result[:, window:] -= totals[:, :-window]
    return result.astype(np.float32)


def days_since(mask):
    positions = np.arange(mask.shape[1], dtype=np.int32)[None, :]
    last = np.maximum.accumulate(np.where(mask, positions, -1), axis=1)
    return np.where(last < 0, -1, positions - last).astype(np.float32)


def augment(mart):
    """Only past/current observations; no target-derived aggregates or future fills."""
    counts = mart.groupby("channel_id", sort=False, observed=True).size()
    if counts.nunique() != 1:
        raise ValueError("Expected complete, channel-sorted daily grid")
    n_channels, n_days = len(counts), int(counts.iloc[0])
    if not mart[["channel_id", "date"]].equals(
        mart[["channel_id", "date"]].sort_values(["channel_id", "date"])
    ):
        raise ValueError("Rows must be sorted by channel/date")
    alarm = mart.alarm_count.to_numpy().reshape(n_channels, n_days) > 0
    observed = mart.events_count.to_numpy().reshape(n_channels, n_days) > 0
    fault = mart.fault_count.to_numpy().reshape(n_channels, n_days) > 0
    for name, mask in [("alarm", alarm), ("observed", observed), ("fault", fault)]:
        mart[f"days_since_{name}"] = days_since(mask).ravel()
    for window in [3, 7, 28, 90]:
        alarm_days = rolling_sum(alarm, window)
        observed_days = rolling_sum(observed, window)
        mart[f"alarm_days_{window}d"] = alarm_days.ravel()
        mart[f"observed_days_{window}d"] = observed_days.ravel()
        mart[f"alarm_day_rate_{window}d"] = (
            alarm_days / np.minimum(np.arange(n_days) + 1, window)[None, :]
        ).ravel().astype(np.float32)
    first = np.maximum.accumulate(observed, axis=1)
    mart["seen_before_or_today"] = first.ravel().astype(np.int8)
    mart["historical_alarm_day_rate"] = (
        np.cumsum(alarm, axis=1) / (np.arange(n_days)[None, :] + 1)
    ).ravel().astype(np.float32)
    mart["alarm_trend_7_28"] = mart.alarm_day_rate_7d - mart.alarm_day_rate_28d
    mart["events_trend_1_7"] = (mart.events_count / (1 + mart.events_count_sum_7d / 7)).astype("float32")
    # Known object membership is context, not a claim of physical adjacency/causality.
    key = [mart.object_id, mart.date]
    for col in ["alarm_count", "fault_count", "no_power_count", "events_count"]:
        totals = mart[col].groupby(key, observed=True).transform("sum")
        mart[f"other_object_{col}"] = (totals - mart[col]).astype("float32")
    mart["channel_category"] = mart.channel_id.astype("category")
    mart["annual_sin"] = np.sin(2 * np.pi * mart.date.dt.dayofyear / 365.25).astype("float32")
    mart["annual_cos"] = np.cos(2 * np.pi * mart.date.dt.dayofyear / 365.25).astype("float32")
    return mart


def choose_threshold(y, score):
    p, r, t = precision_recall_curve(y, score)
    f = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-15)
    best = int(np.argmax(f))
    eligible = np.where((p[:-1] > .7) & (r[:-1] > .5))[0]
    precise = np.where(p[:-1] > .7)[0]
    precise_threshold = float(t[precise[np.argmax(r[precise])]]) if len(precise) else None
    return float(t[best]), {
        "max_f1": float(f[best]), "precision_at_max_f1": float(p[best]),
        "recall_at_max_f1": float(r[best]), "requirement_achievable": bool(len(eligible)),
        "max_recall_at_precision_gt_0_7": float(r[:-1][p[:-1] > .7].max(initial=0)),
        "threshold_precision_gt_0_7": precise_threshold,
    }


def metrics(y, score, threshold):
    y = np.asarray(y); pred = np.asarray(score) >= threshold
    tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    return {"rows": len(y), "positives": int(y.sum()), "positive_rate": float(y.mean()),
            "average_precision": float(average_precision_score(y, score)),
            "roc_auc": float(roc_auc_score(y, score)), "threshold": threshold,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-15),
            "true_positive": tp, "false_positive": fp, "false_negative": fn}


def frame(mart, indexes, features, categorical, catboost=False):
    result = mart.loc[indexes, features].copy()
    if catboost:
        for col in categorical:
            if col in result:
                result[col] = result[col].astype(str)
        for col in result.select_dtypes(include="number"):
            result[col] = result[col].replace([np.inf, -np.inf], np.nan).fillna(-999999)
    return result


def run(args):
    for folder in [OUT, OUT / "models", OUT / "predictions"]:
        folder.mkdir(parents=True, exist_ok=True)
    mart = prepare()
    if args.prepare_only:
        return
    baseline_features, base_cats = legacy.feature_columns(mart)
    old = lgb.Booster(model_file=str(legacy.MODELS / "lightgbm_baseline.txt"))
    if baseline_features != old.feature_name():
        raise ValueError("Baseline feature schema mismatch")
    # One-day label embargo. Keep legacy validation/test rows for like-for-like reporting.
    train_idx = mart.index[(mart.split == "train") & (mart.date < "2025-09-30")]
    valid_idx = mart.index[(mart.split == "valid") & (mart.date < "2025-12-31")]
    legacy_valid_idx = mart.index[mart.split == "valid"]
    test_idx = legacy.sample_eval(mart[mart.split == "test"], 1_200_000, 42).index
    # Sample directly on row identities; the legacy sampler resets indexes.
    positives = train_idx[mart.loc[train_idx, TARGET].to_numpy() == 1]
    negatives = train_idx[mart.loc[train_idx, TARGET].to_numpy() == 0]
    rng = np.random.default_rng(42)
    n_pos = min(len(positives), int(args.train_rows * .45))
    n_neg = min(len(negatives), args.train_rows - n_pos)
    selected_pos = rng.choice(positives, n_pos, replace=False)
    selected_neg = rng.choice(negatives, n_neg, replace=False)
    sampled_idx = np.concatenate([selected_pos, selected_neg]); rng.shuffle(sampled_idx)
    y_train = mart.loc[sampled_idx, TARGET].to_numpy()
    # Inverse selection weights recover original class prior, without double balancing.
    weights = np.where(y_train == 1, len(positives)/n_pos, len(negatives)/n_neg)
    weights /= weights.mean()
    y_valid = mart.loc[valid_idx, TARGET].to_numpy()
    log(f"Train={len(sampled_idx):,}, valid={len(valid_idx):,}; baseline trees={old.num_trees()}")
    baseline_valid = old.predict(mart.loc[legacy_valid_idx, baseline_features], num_threads=6)
    reference = pd.read_parquet(legacy.TABLES / "predictions_sample.parquet")
    reference = reference[reference.split == "valid"]
    check = mart.loc[legacy_valid_idx, ["channel_id", "date", TARGET]].copy()
    check["reproduced_score"] = baseline_valid
    check = check.merge(reference, on=["channel_id", "date", TARGET], validate="one_to_one")
    difference = float(np.max(np.abs(check.reproduced_score - check.score)))
    if len(check) != len(reference) or difference > 1e-6:
        raise ValueError(f"Archived baseline reproduction failed: rows={len(check)}, max_diff={difference}")
    log(f"Baseline validation reproduced, maximum score difference {difference:.3g}")
    del reference, check
    mart = augment(mart)
    additions = [c for c in mart if c not in baseline_features + ["date", "channel_id", TARGET, "split", "value_sum", "value_sumsq"]]
    enhanced_features = [c for c in baseline_features if c not in ["year", "month"]] + additions
    cats = base_cats + ["channel_category"]
    trials = [
        ("lgb_corrected", baseline_features, base_cats, "lgb"),
        ("lgb_history", enhanced_features, cats, "lgb"),
        ("catboost_history", enhanced_features, cats, "catboost"),
    ]
    results = []; trained = {}
    for name, features, categorical, kind in trials:
        if args.skip_catboost and kind == "catboost":
            continue
        log(f"Training {name}: {len(features)} features")
        start = time.time()
        x_train = frame(mart, sampled_idx, features, categorical, kind == "catboost")
        x_valid = frame(mart, valid_idx, features, categorical, kind == "catboost")
        if kind == "lgb":
            model = lgb.LGBMClassifier(objective="binary", metric="average_precision",
                n_estimators=1000, learning_rate=.04, num_leaves=31, min_child_samples=150,
                reg_lambda=5, colsample_bytree=.85, subsample=.85, subsample_freq=1,
                random_state=42, n_jobs=6, verbosity=-1)
            model.fit(x_train, y_train, sample_weight=weights,
                eval_set=[(x_valid, y_valid)], categorical_feature=categorical,
                callbacks=[lgb.early_stopping(75, first_metric_only=True), lgb.log_evaluation(100)])
            model.booster_.save_model(str(OUT / "models" / f"{name}.txt"))
            iterations = model.best_iteration_
        else:
            model = CatBoostClassifier(iterations=800, depth=6, learning_rate=.06,
                loss_function="Logloss", eval_metric="PRAUC", l2_leaf_reg=5,
                random_seed=42, thread_count=6, allow_writing_files=False)
            model.fit(x_train, y_train, sample_weight=weights, cat_features=categorical,
                eval_set=(x_valid, y_valid), early_stopping_rounds=75, verbose=100)
            model.save_model(str(OUT / "models" / f"{name}.cbm"))
            iterations = model.tree_count_
        score = model.predict_proba(x_valid)[:, 1]
        threshold, operating = choose_threshold(y_valid, score)
        row = {"model": name, "seconds": time.time()-start, "iterations": int(iterations),
               **metrics(y_valid, score, threshold), **operating}
        results.append(row); trained[name] = (model, features, categorical, kind, threshold)
        pd.DataFrame(results).to_csv(OUT / "validation_comparison.csv", index=False)
        log(json.dumps(row))
        del x_train, x_valid; gc.collect()
    winner = max(results, key=lambda r: r["average_precision"])["model"]
    model, features, categorical, kind, threshold = trained[winner]
    # Freeze decision before touching test scores/labels.
    selection = {"selected_model": winner, "criterion": "validation average_precision",
        "features": features, "categorical_features": categorical, "threshold": threshold,
        "train_feature_end": "2025-09-29", "validation_feature_end": "2025-12-30",
        "target": TARGET, "prediction_time": "end of feature date",
        "raw_fingerprint": fingerprint(), "baseline_validation_max_score_difference": difference,
        "train_rows": len(sampled_idx), "sampling": "positive/negative sampling, inverse probability weights",
        "unobserved_policy": "legacy full grid, no record remains zero for comparison; not confirmed healthy",
        "test_policy": "legacy deterministic 1,200,000 sample, random_state=42"}
    selection["threshold_precision_gt_0_7"] = next(r for r in results if r["model"] == winner)["threshold_precision_gt_0_7"]
    (OUT / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2))
    log(f"Selected {winner}; selection saved. Evaluating held-out test.")
    comparison = []
    operating_points = []
    old_threshold = json.loads((legacy.TABLES / "baseline_metadata.json").read_text())["best_threshold_valid_f1"]
    for split, indexes in [("valid_legacy", legacy_valid_idx), ("test", test_idx)]:
        y = mart.loc[indexes, TARGET].to_numpy()
        old_score = old.predict(mart.loc[indexes, baseline_features], num_threads=6)
        x = frame(mart, indexes, features, categorical, kind == "catboost")
        score = model.predict_proba(x)[:, 1]
        comparison.append({"model": "archived_baseline", "split": split, **metrics(y, old_score, old_threshold)})
        comparison.append({"model": winner, "split": split, **metrics(y, score, threshold)})
        precise_threshold = selection["threshold_precision_gt_0_7"]
        if precise_threshold is not None:
            operating_points.append({"model": winner, "split": split, "policy": "precision_gt_0.7_on_validation",
                                     **metrics(y, score, precise_threshold)})
        pred = mart.loc[indexes, ["date", "channel_id", TARGET, "events_count", "alarm_count"]].copy()
        pred["baseline_score"] = old_score; pred["score"] = score
        pred.to_parquet(OUT / "predictions" / f"{split}.parquet", index=False)
        if split == "test":
            smoke_indexes = indexes[:256]
            mart.loc[smoke_indexes, ["channel_id", "date"] + features].to_parquet(
                CACHE / "scoring_smoke.parquet", index=False)
            from predict_risk import score_features
            smoke = score_features(mart.loc[smoke_indexes], OUT)
            np.testing.assert_allclose(smoke.score.to_numpy(), score[:256], rtol=1e-10, atol=1e-12)
            (OUT / "inference_check.json").write_text(json.dumps({"rows": 256, "passed": True,
                "max_score_difference": float(np.max(np.abs(smoke.score.to_numpy()-score[:256])))}))
        log(json.dumps(comparison[-1]))
        del x, pred; gc.collect()
    pd.DataFrame(comparison).to_csv(OUT / "comparison.csv", index=False)
    pd.DataFrame(operating_points).to_csv(OUT / "operating_points.csv", index=False)
    importance = model.feature_importances_ if kind == "lgb" else model.get_feature_importance()
    pd.DataFrame({"feature": features, "importance": importance}).sort_values("importance", ascending=False).to_csv(OUT / "feature_importance.csv", index=False)
    log("Finished. No test-based model reselection.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--train-rows", type=int, default=700_000)
    parser.add_argument("--skip-catboost", action="store_true")
    run(parser.parse_args())
