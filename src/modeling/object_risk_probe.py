#!/usr/bin/env python3
"""Separate object-day investigation; any registered alarm in the +24…+48h window.

Only January-November 2025 is read. This is not a replacement of channel-level
evaluation, an equipment-failure label, or a production model change.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/object_risk_probe"
CACHE = ROOT / "data/interim"
TARGET = "target_any_object_alarm_d_plus_2"
COUNTS = ["events_count", "alarm_count", "fault_count", "no_power_count", "undefined_count",
          "numeric_count", "text_count", "off_count", "on_count", "movement_detected_count", "open_count"]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(end_date="2025-11-30"):
    # The historical probe's default still excludes December in the reader.
    # A frozen forward check may explicitly request a later feature horizon.
    end_date = pd.Timestamp(end_date)
    raw = pd.concat([
        pd.read_parquet(CACHE / f"daily_{year}.parquet", columns=["channel_id", "date", *COUNTS],
                        filters=[("date", ">=", pd.Timestamp("2025-01-01")),
                                 ("date", "<=", end_date)])
        for year in range(2025, end_date.year + 1)
    ], ignore_index=True)
    meta = pd.read_csv(ROOT / "data/raw/справочник_каналов_датчиков.csv", dtype=str).rename(
        columns={"ид_канала_данных": "channel_id", "ид_объект": "object_id"})
    objects = pd.read_csv(ROOT / "data/raw/справочник_объектов_диспетчер.csv", dtype=str).rename(
        columns={"ид_объект": "object_id", "вид_объекта": "object_kind", "родитель": "parent_id"})
    return prepare_frames(raw, meta, objects, end_date)


def prepare_frames(raw, meta, objects, end_date):
    """Same feature logic on caller-supplied, already available daily records.

    File-based historical training still uses prepare(). A serving adapter can
    supply a strictly truncated snapshot without opening future evaluation data.
    """
    end_date = pd.Timestamp(end_date)
    if raw.date.isna().any() or raw.date.gt(end_date).any():
        raise ValueError("Daily frame contains unknown or future dates")
    if raw.duplicated(["channel_id", "date"]).any():
        raise ValueError("Duplicate channel-day records")
    if meta.channel_id.duplicated().any() or objects.object_id.duplicated().any():
        raise ValueError("Ambiguous metadata identifiers")
    eligible_meta = meta[meta.object_id.isin(objects.object_id)]
    identities = sorted(eligible_meta.object_id.unique())
    joined = raw.merge(eligible_meta[["channel_id", "object_id"]], on="channel_id", how="left", validate="many_to_one")
    unknown = joined.object_id.isna()
    coverage = {"catalog_objects_with_channels": len(identities),
                "excluded_unknown_channel_days": int(unknown.sum()),
                "excluded_alarm_channel_days": int(joined.loc[unknown, "alarm_count"].gt(0).sum()),
                "included_channel_days": int((~unknown).sum())}
    joined = joined.loc[~unknown].copy()
    joined["observed_channels"] = joined.events_count.gt(0).astype("int16")
    joined["alarm_channels"] = joined.alarm_count.gt(0).astype("int16")
    joined["fault_channels"] = joined.fault_count.gt(0).astype("int16")
    joined["no_power_channels"] = joined.no_power_count.gt(0).astype("int16")
    sums = COUNTS + ["observed_channels", "alarm_channels", "fault_channels", "no_power_channels"]
    daily = joined.groupby(["object_id", "date"], observed=True)[sums].sum().reset_index()
    dates = pd.date_range("2025-01-01", end_date)
    grid = pd.MultiIndex.from_product([identities, dates], names=["object_id", "date"]).to_frame(index=False)
    mart = grid.merge(daily, on=["object_id", "date"], how="left", validate="one_to_one")
    mart[sums] = mart[sums].fillna(0).astype("float32")
    counts = eligible_meta.groupby("object_id").channel_id.nunique()
    mart["catalog_channels"] = mart.object_id.map(counts).astype("float32")
    mart = mart.merge(objects[["object_id", "object_kind", "parent_id"]], on="object_id", how="left", validate="many_to_one")
    mart = mart.sort_values(["object_id", "date"]).reset_index(drop=True)
    mart["alarm_today"] = mart.alarm_channels.gt(0).astype("float32")
    mart["observed_today"] = mart.observed_channels.gt(0).astype("float32")
    mart["observed_fraction"] = mart.observed_channels / mart.catalog_channels
    mart["alarm_channel_fraction"] = mart.alarm_channels / mart.observed_channels.replace(0, np.nan)
    for column in ["object_id", "object_kind", "parent_id"]:
        mart[column] = mart[column].fillna("unknown").astype("category")
    groups = mart.groupby("object_id", observed=True, sort=False)
    for column in ["alarm_today", "alarm_channels", "observed_channels", "fault_channels", "no_power_channels", "events_count"]:
        for days in [1, 2, 7, 14]:
            mart[f"{column}_lag{days}"] = groups[column].shift(days).astype("float32")
        for window in [3, 7, 28]:
            mart[f"{column}_mean{window}"] = groups[column].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True).astype("float32")
    ordinal = groups.cumcount().to_numpy()
    mart["historical_alarm_frequency"] = groups.alarm_today.cumsum() / (ordinal + 1)
    a = mart.alarm_today.to_numpy().reshape(len(identities), len(dates)).astype(bool)
    position = np.arange(len(dates))[None, :]
    last_clear = np.maximum.accumulate(np.where(~a, position, -1), axis=1)
    last_alarm = np.maximum.accumulate(np.where(a, position, -1), axis=1)
    mart["alarm_streak"] = np.where(a, position-last_clear, 0).ravel().astype("float32")
    mart["days_since_alarm"] = np.where(last_alarm < 0, -1, position-last_alarm).ravel().astype("float32")
    mart["target_dayofweek"] = (mart.date + pd.Timedelta(days=2)).dt.dayofweek.astype("int8")
    mart["annual_sin"] = np.sin(2*np.pi*mart.date.dt.dayofyear/365.25).astype("float32")
    mart["annual_cos"] = np.cos(2*np.pi*mart.date.dt.dayofyear/365.25).astype("float32")
    # Rebuild grouping after additions; dates are explicitly checked, never inferred from row position alone.
    groups = mart.groupby("object_id", observed=True, sort=False)
    future_date = groups.date.shift(-2)
    available = future_date.notna()
    assert future_date[available].eq(mart.loc[available, "date"]+pd.Timedelta(days=2)).all()
    mart[TARGET] = groups.alarm_today.shift(-2)
    coverage["objects_with_observations_by_nov30"] = int(mart.loc[mart.observed_today.eq(1) & mart.date.le("2025-11-30"), "object_id"].nunique())
    return mart, future_date, coverage


def threshold_and_feasibility(y, score):
    p, r, t = precision_recall_curve(y, score)
    p, r = p[:-1], r[:-1]
    f1 = 2*p*r/np.maximum(p+r, 1e-15)
    best = int(np.argmax(f1))
    feasible = np.flatnonzero((p > .7) & (r > .5))
    reference_threshold = float(t[feasible[np.argmax(r[feasible])]]) if len(feasible) else None
    return float(t[best]), {"engineering_reference_70_50_achievable": bool(len(feasible)),
                           "threshold_status": "Engineering reference only; original task section 9 sets no mandatory 70/50 thresholds",
                           "max_recall_at_precision_gt_0_7": float(r[p > .7].max(initial=0)),
                           "engineering_reference_threshold": reference_threshold}


def metrics(name, cohort, y, score, threshold, days):
    pred = score >= threshold
    tp = int(((y == 1) & pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    p = tp/max(tp+fp, 1)
    r = tp/max(tp+fn, 1)
    return {"model": name, "cohort": cohort, "rows": len(y), "positives": int(y.sum()),
            "prevalence": float(y.mean()), "average_precision": float(average_precision_score(y, score)),
            "roc_auc": float(roc_auc_score(y, score)), "precision": p, "recall": r,
            "f1": 2*p*r/max(p+r, 1e-15), "threshold": threshold,
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "calendar_days": days, "false_alerts_per_calendar_day": fp/days,
            "all_alerts_per_calendar_day": int(pred.sum())/days}


def main():
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    mart, label_date, coverage = prepare()
    train = mart.index[mart.date.le("2025-09-28")]
    tune = mart.index[mart.date.between("2025-10-01", "2025-11-28")]
    assert mart.loc[train, TARGET].notna().all() and mart.loc[tune, TARGET].notna().all()
    assert label_date.loc[train].max() <= pd.Timestamp("2025-09-30")
    assert label_date.loc[tune].max() <= pd.Timestamp("2025-11-30")
    features = [c for c in mart if c not in [TARGET, "date"]]
    cats = ["object_id", "object_kind", "parent_id"]
    yt = mart.loc[train, TARGET].to_numpy(dtype="int8")
    yv = mart.loc[tune, TARGET].to_numpy(dtype="int8")
    model = lgb.LGBMClassifier(objective="binary", metric="average_precision", n_estimators=400,
        num_leaves=15, learning_rate=.04, min_child_samples=30, reg_lambda=10,
        colsample_bytree=.85, random_state=84, n_jobs=2, verbosity=-1)
    model.fit(mart.loc[train, features], yt, eval_set=[(mart.loc[tune, features], yv)],
        categorical_feature=cats, callbacks=[lgb.early_stopping(40, first_metric_only=True), lgb.log_evaluation(100)])
    # Historical baseline is a frozen per-object training target rate, never a tune-label aggregate.
    frequency = mart.loc[train].groupby("object_id", observed=True)[TARGET].mean()
    scores = {
        "always_positive": np.ones(len(tune)),
        "persistence_alarm_today": mart.loc[tune, "alarm_today"].to_numpy(),
        "frozen_object_frequency": mart.loc[tune, "object_id"].map(frequency).astype(float).to_numpy(),
        "lightgbm_object": model.predict_proba(mart.loc[tune, features], num_threads=2)[:, 1],
    }
    thresholds, feasibility = {}, {}
    rows = []
    days = mart.loc[tune, "date"].nunique()
    quiet = mart.loc[tune, "alarm_today"].eq(0).to_numpy()
    for name, score in scores.items():
        chosen, facts = threshold_and_feasibility(yv, score)
        threshold = .5 if name in ["always_positive", "persistence_alarm_today"] else chosen
        thresholds[name] = threshold
        feasibility[name] = facts
        rows.append(metrics(name, "all_object_days", yv, score, threshold, days))
        rows.append(metrics(name, "no_alarm_on_feature_day", yv[quiet], score[quiet], threshold, days))
        if facts["engineering_reference_threshold"] is not None and name in ["frozen_object_frequency", "lightgbm_object"]:
            rows.append(metrics(name+"_engineering_reference_point", "all_object_days", yv, score,
                                facts["engineering_reference_threshold"], days))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "metrics.csv", index=False)
    dump(OUT / "feasibility.json", feasibility)
    prediction = mart.loc[tune, ["object_id", "date", "alarm_today", TARGET]].copy()
    for name, score in scores.items():
        prediction[name] = score
    prediction.to_parquet(CACHE / "object_risk_tuning_predictions.parquet", index=False)
    model_path = ROOT / "models/object_risk_probe.txt"
    model_path.parent.mkdir(exist_ok=True)
    model.booster_.save_model(str(model_path))
    pd.DataFrame({"feature": features, "gain": model.booster_.feature_importance("gain")}).sort_values(
        "gain", ascending=False).to_csv(OUT / "feature_importance.csv", index=False)
    temporal = {"unit": "object-day, not channel-day", "target": TARGET,
        "meaning": "Any alarm-flagged registered record in at least one channel; not physical failure",
        "issue_time": "D+1 00:00", "label_interval": "[D+2 00:00,D+3 00:00)", "minimum_lead_hours": 24,
        "training_feature_end": str(mart.loc[train,"date"].max()), "training_label_last_day": str(label_date.loc[train].max()),
        "tuning_feature_start": str(mart.loc[tune,"date"].min()), "tuning_feature_end": str(mart.loc[tune,"date"].max()),
        "tuning_label_last_day": str(label_date.loc[tune].max()), "december_used": False, "test_2026_used": False,
        "exact_object_calendar_shift_verified": True, "training_rows": len(train), "tuning_rows": len(tune),
        "training_prevalence": float(yt.mean()), "tuning_prevalence": float(yv.mean()),
        "quiet_today_tuning_rows": int(quiet.sum()), "quiet_today_tuning_prevalence": float(yv[quiet].mean()),
        "quiet_subset_note": "No alarm on feature day only; an alarm may start in the unscored intermediate day D+1",
        "threshold_policy": "Fixed .5 for always-positive/persistence; max-F1 on the same tuning for model and historical-frequency baseline",
        "selection_limitation": "Early stopping and threshold tuning share Oct-Nov; preliminary validation only",
        "coverage": coverage, "elapsed_seconds": time.monotonic()-start}
    dump(OUT / "temporal_checks.json", temporal)
    dump(OUT / "configuration.json", {"parameters": model.get_params(), "best_iteration": model.best_iteration_,
        "features": features, "categorical": cats, "thresholds": thresholds,
        "model_path": str(model_path.relative_to(ROOT)), "model_sha256": checksum(model_path),
        "code_sha256": checksum(Path(__file__)),
        "daily_2025_sha256": checksum(CACHE / "daily_2025.parquet"),
        "meta_sha256": checksum(ROOT / "data/raw/справочник_каналов_датчиков.csv"),
        "objects_sha256": checksum(ROOT / "data/raw/справочник_объектов_диспетчер.csv")})
    print(table.to_string(index=False), flush=True)
    print(json.dumps(temporal, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
