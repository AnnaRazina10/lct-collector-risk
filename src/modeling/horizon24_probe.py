#!/usr/bin/env python3
"""Preliminary 2025-only probe: alarm occurrence in the +24 to +48h window.

Issue time is the midnight after feature day D; the target is an alarm on D+2.
Training labels finish by Sep30; tuning labels finish by Nov30. No December
calibration or 2026 evaluation is performed. This is a different target from v2.
"""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/horizon24_probe"
CACHE = ROOT / "data/interim"
TARGET = "target_alarm_lead24_window24"
CATS = ["engineering_system", "sensor_type", "object_id", "object_kind", "channel_category"]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_target(mart):
    """Shift only within a channel and require the exact calendar day D+2."""
    if mart.duplicated(["channel_id", "date"]).any():
        raise ValueError("Duplicate channel/day identities")
    if not mart[["channel_id", "date"]].equals(mart[["channel_id", "date"]].sort_values(["channel_id", "date"])):
        raise ValueError("Unsorted channel/day rows")
    groups = mart.groupby("channel_id", observed=True, sort=False)
    label_date = groups.date.shift(-2)
    future_alarm = groups.alarm_count.shift(-2)
    available = label_date.notna()
    if not label_date[available].eq(mart.loc[available, "date"] + pd.Timedelta(days=2)).all():
        raise ValueError("Nonconsecutive calendar days: row shift is not a D+2 target")
    target = future_alarm.gt(0).astype("float32").where(available)
    return target, label_date


def operating(y, score):
    precision, recall, thresholds = precision_recall_curve(y, score)
    p, r = precision[:-1], recall[:-1]
    f1 = 2 * p * r / np.maximum(p + r, 1e-15)
    best = int(np.argmax(f1))
    precise = p > .7
    return float(thresholds[best]), {
        "engineering_reference_70_50_achievable_on_tuning": bool(((p > .7) & (r > .5)).any()),
        "threshold_status": "Engineering reference only; original task section 9 sets no mandatory 70/50 thresholds",
        "max_recall_at_precision_gt_0_7": float(r[precise].max(initial=0)),
        "max_precision_at_recall_gt_0_5": float(p[r > .5].max(initial=0)),
    }, (precision, recall, thresholds)


def metric(y, score, threshold):
    pred = score >= threshold
    tp = int(((y == 1) & pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {"rows": len(y), "positives": int(y.sum()), "positive_rate": float(y.mean()),
            "average_precision": float(average_precision_score(y, score)),
            "roc_auc": float(roc_auc_score(y, score)), "threshold": threshold,
            "precision": p, "recall": r, "f1": 2*p*r/max(p+r, 1e-15),
            "true_positive": tp, "false_positive": fp, "false_negative": fn}


def main():
    start = time.monotonic()
    pa.set_cpu_count(2)
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)
    source = CACHE / "advanced_mart.parquet"
    prior_path = CACHE / "channel_prior_frozen_v2.parquet"
    intraday_path = CACHE / "intraday_2025.parquet"
    # Exclude the legacy label at the I/O boundary: Nov30's legacy label is Dec1.
    excluded = {"split", "value_sum", "value_sumsq", "year", "month"}
    columns = [c for c in pq.ParquetFile(source).schema_arrow.names
               if c not in excluded and not c.startswith("target_")]
    filters = [("date", ">=", pd.Timestamp("2025-01-01")),
               ("date", "<=", pd.Timestamp("2025-11-30"))]
    mart = pd.read_parquet(source, columns=columns, filters=filters)
    mart = mart.merge(pd.read_parquet(prior_path), on="channel_id", how="left", validate="many_to_one", sort=False)
    mart = mart.merge(pd.read_parquet(intraday_path, filters=filters),
                      on=["channel_id", "date"], how="left", validate="one_to_one", sort=False)
    for c in mart:
        if c.startswith("intra_"):
            missing = -1 if ("minute" in c or "is_alarm" in c) else 0
            mart[c] = mart[c].fillna(missing)
    mart = mart.sort_values(["channel_id", "date"]).reset_index(drop=True)
    features = [c for c in mart if c not in {"date", "channel_id"}]
    for c in features:
        if c.startswith("target_") or c in excluded:
            raise AssertionError("Ineligible feature: " + c)
        if c not in CATS and not pd.api.types.is_numeric_dtype(mart[c]):
            raise ValueError("Unexpected non-numeric feature: " + c)
    mart[TARGET], label_date = make_target(mart)
    train_idx = mart.index[mart.date.le("2025-09-28")]
    tune_idx = mart.index[mart.date.between("2025-10-01", "2025-11-28")]
    assert mart.loc[train_idx, TARGET].notna().all() and mart.loc[tune_idx, TARGET].notna().all()
    assert label_date.loc[train_idx].max() <= pd.Timestamp("2025-09-30")
    assert label_date.loc[tune_idx].max() <= pd.Timestamp("2025-11-30")
    assert mart.date.min() >= pd.Timestamp("2025-01-01") and mart.date.max() <= pd.Timestamp("2025-11-30")
    temporal = {
        "target": TARGET, "description": "Any alarm-flagged record on calendar day D+2; not confirmed equipment failure",
        "issue_time": "D+1 00:00 in source calendar", "label_interval": "[D+2 00:00, D+3 00:00)",
        "minimum_lead_hours": 24, "window_hours": 24,
        "read_feature_start": str(mart.date.min()), "read_feature_end": str(mart.date.max()),
        "training_features_end": str(mart.loc[train_idx, "date"].max()),
        "training_label_last_day": str(label_date.loc[train_idx].max()),
        "tuning_features_start": str(mart.loc[tune_idx, "date"].min()),
        "tuning_features_end": str(mart.loc[tune_idx, "date"].max()),
        "tuning_label_last_day": str(label_date.loc[tune_idx].max()),
        "exact_channel_calendar_shift_verified": True, "duplicate_channel_days": 0,
        "legacy_target_read": False, "december_calibration_used": False, "test_2026_used": False,
        "selection_limitation": "Early stopping and threshold selection share Oct-Nov tuning; preliminary validation only",
    }
    dump(OUT / "temporal_checks.json", temporal)
    y_full = mart.loc[train_idx, TARGET].to_numpy()
    positive = train_idx[y_full == 1]
    negative = train_idx[y_full == 0]
    rng = np.random.default_rng(84)
    cap = 600_000
    pos = rng.choice(positive, min(len(positive), cap // 3), replace=False)
    neg = rng.choice(negative, min(len(negative), cap-len(pos)), replace=False)
    sampled = np.r_[pos, neg]
    rng.shuffle(sampled)
    yt = mart.loc[sampled, TARGET].to_numpy(dtype="int8")
    yv = mart.loc[tune_idx, TARGET].to_numpy(dtype="int8")
    weights = np.where(yt == 1, len(positive)/len(pos), len(negative)/len(neg))
    weights /= weights.mean()
    sample_path = CACHE / "horizon24_training_sample.parquet"
    sample = mart.loc[sampled, ["channel_id", "date", TARGET]].copy()
    sample["inverse_sampling_weight"] = weights
    sample.to_parquet(sample_path, index=False)
    x = mart.loc[sampled, features].copy()
    xv = mart.loc[tune_idx, features].copy()
    tuning_keys = mart.loc[tune_idx, ["channel_id", "date", TARGET]].copy()
    del mart, sample, label_date
    gc.collect()
    parameters = dict(objective="binary", metric="average_precision", n_estimators=600,
                      learning_rate=.035, num_leaves=31, min_child_samples=120,
                      reg_lambda=10, colsample_bytree=.85, subsample=.85,
                      subsample_freq=1, random_state=84, n_jobs=2, verbosity=-1)
    print(f"[horizon24] train={len(yt):,}, tune={len(yv):,}, features={len(features)}", flush=True)
    model = lgb.LGBMClassifier(**parameters)
    model.fit(x, yt, sample_weight=weights, eval_set=[(xv, yv)], categorical_feature=CATS,
              callbacks=[lgb.early_stopping(70, first_metric_only=True), lgb.log_evaluation(100)])
    score = model.predict_proba(xv, num_threads=2)[:, 1]
    model_path = ROOT / "models/horizon24_probe.txt"
    model.booster_.save_model(str(model_path))
    threshold, feasibility, curve = operating(yv, score)
    measured = metric(yv, score, threshold)
    result = {"status": "preliminary_tuning_only", "model": "LightGBM31", "target": TARGET,
              "metrics": measured, "feasibility": feasibility, "best_iteration": model.best_iteration_,
              "training_population_rows": len(train_idx), "training_population_positives": len(positive),
              "training_sample_rows": len(sampled), "elapsed_seconds": time.monotonic()-start}
    dump(OUT / "metrics.json", result)
    dump(OUT / "configuration.json", {"parameters": parameters, "features": features, "categorical": CATS,
         "early_stopping_rounds": 70, "threshold": threshold, "model_path": str(model_path.relative_to(ROOT)),
         "model_sha256": sha256(model_path), "sample_path": str(sample_path.relative_to(ROOT)),
         "sample_sha256": sha256(sample_path), "prior_sha256": sha256(prior_path),
         "intraday_sha256": sha256(intraday_path), "code_sha256": sha256(Path(__file__)),
         "source_file": str(source.relative_to(ROOT)), "source_sha256": sha256(source)})
    tuning_keys["score"] = score
    tuning_keys.to_parquet(CACHE / "horizon24_tuning_predictions.parquet", index=False)
    np.savez_compressed(CACHE / "horizon24_precision_recall_curve.npz",
                        precision=curve[0], recall=curve[1], thresholds=curve[2])
    pd.DataFrame({"feature": features, "gain": model.booster_.feature_importance("gain")}).sort_values(
        "gain", ascending=False).to_csv(OUT / "feature_importance.csv", index=False)
    reloaded = lgb.Booster(model_file=str(model_path))
    max_difference = float(np.max(np.abs(reloaded.predict(xv.iloc[:256], num_threads=2)-score[:256])))
    if max_difference > 1e-12:
        raise AssertionError("Saved model predictions changed")
    dump(OUT / "reload_check.json", {"rows": 256, "maximum_prediction_difference": max_difference})
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
