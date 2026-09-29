#!/usr/bin/env python3
"""Export frozen object-risk predictions using a strictly truncated feature history.

Archive outcomes are attached only after inference, for a separate explicit UI
action. No training, threshold selection, weather enrichment or model change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import object_risk_probe as source

ROOT = source.ROOT
FEATURE_DATE = "2026-06-28"
LABELS = {
    "object_id": "Исторический профиль объекта",
    "parent_id": "Группа объекта в справочнике",
    "object_kind": "Тип объекта",
    "historical_alarm_frequency": "Доля дней с тревогой в прошлой истории",
    "alarm_today_mean28": "Доля дней с тревогой за 28 дней",
    "alarm_today_mean7": "Доля дней с тревогой за неделю",
    "alarm_today_mean3": "Доля дней с тревогой за 3 дня",
    "alarm_streak": "Дней подряд с тревожными записями",
    "alarm_channels_mean28": "Среднее число тревоживших каналов за 28 дней",
    "alarm_channels_mean7": "Среднее число тревоживших каналов за неделю",
    "events_count_mean28": "Среднее число записей за 28 дней",
    "target_dayofweek": "День недели прогнозируемого окна",
    "annual_sin": "Сезонность по календарю",
    "annual_cos": "Сезонность по календарю",
    "alarm_today": "Тревожные записи в день признаков",
    "alarm_count": "Число тревожных записей в день признаков",
    "observed_channels": "Каналы с записями в день признаков",
    "catalog_channels": "Каналы объекта в справочнике",
    "days_since_alarm": "Дней с последней зарегистрированной тревоги",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forecast_times(feature_date):
    day = pd.Timestamp(feature_date).normalize().tz_localize("Europe/Moscow")
    issue = day + pd.Timedelta(days=1)
    start = day + pd.Timedelta(days=2)
    end = day + pd.Timedelta(days=3)
    assert (start-issue).total_seconds() == 86400
    return {"feature_date": str(day.date()), "feature_cutoff": (issue-pd.Timedelta(seconds=1)).isoformat(),
            "issue_time": issue.isoformat(), "forecast_start": start.isoformat(),
            "forecast_end": end.isoformat(), "minimum_lead_hours": 24, "window_hours": 24,
            "timezone": "Europe/Moscow"}


def prediction_frame(feature_date, cfg):
    """Do not read records after the feature cutoff, even for deriving the label."""
    cutoff = pd.Timestamp(feature_date).normalize()
    mart, _, coverage = source.prepare(str(cutoff.date()))
    if mart.date.max() != cutoff or (mart.date > cutoff).any():
        raise ValueError("Feature history extends beyond the requested cutoff")
    latest = mart.loc[mart.date.eq(cutoff)].copy()
    if latest.empty or latest.object_id.duplicated().any():
        raise ValueError("Expected one current row per object")
    if not latest[source.TARGET].isna().all():
        raise ValueError("Future outcomes are unexpectedly present in the truncated history")
    if source.TARGET in cfg["features"] or "date" in cfg["features"]:
        raise ValueError("Outcome or timestamp included in model features")
    # target_dayofweek is a known calendar feature, not an observed future outcome.
    expected_weekday = (cutoff+pd.Timedelta(days=2)).dayofweek
    if not latest.target_dayofweek.eq(expected_weekday).all():
        raise ValueError("Forecast calendar feature mismatch")
    recent = mart.loc[mart.date.between(cutoff-pd.Timedelta(days=6), cutoff)]
    observed_days = recent.groupby("object_id", observed=True).observed_today.sum()
    latest["observed_days_7d"] = latest.object_id.map(observed_days).astype(int)
    return latest, coverage


def value_text(value):
    if pd.isna(value):
        return "Нет наблюдения"
    if isinstance(value, (float, np.floating)):
        return f"{value:.3g}"
    return str(value)


def export(feature_date=FEATURE_DATE, destination=None):
    started = time.monotonic()
    cfg_path = ROOT / "reports/object_risk_probe/configuration.json"
    cfg = json.loads(cfg_path.read_text())
    model_path = ROOT / cfg["model_path"]
    if sha256(model_path) != cfg["model_sha256"]:
        raise ValueError("Frozen object model checksum changed")
    checked = json.loads((ROOT / "reports/object_forward_check/checks.json").read_text())
    if not checked.get("passed") or checked["model_sha256"] != cfg["model_sha256"]:
        raise ValueError("Object model differs from the verified forward check")
    times = forecast_times(feature_date)
    latest, coverage = prediction_frame(feature_date, cfg)
    model = lgb.Booster(model_file=str(model_path))
    x = latest[cfg["features"]]
    if model.feature_name() != cfg["features"]:
        raise ValueError("Frozen feature schema mismatch")
    # Inference is complete before loading any archive outcome.
    scores = model.predict(x, num_threads=2)
    contributions = model.predict(x, pred_contrib=True, num_threads=2)
    threshold = float(cfg["thresholds"]["lightgbm_object"])
    latest["score"] = scores

    archived = pd.read_parquet(ROOT / "data/interim/object_forward_predictions.parquet",
                               filters=[("date", "==", pd.Timestamp(feature_date))])
    check = latest[["object_id", "date", "score"]].merge(
        archived[["object_id", "date", source.TARGET, "lightgbm_object"]],
        on=["object_id", "date"], how="left", validate="one_to_one")
    if len(check) != len(latest) or check.lightgbm_object.isna().any():
        raise ValueError("Missing rows in the frozen forward-check archive")
    difference = float(np.max(np.abs(check.score-check.lightgbm_object)))
    if difference > 1e-12:
        raise ValueError("Truncating future history changed a frozen prediction")
    actual = dict(zip(check.object_id.astype(str), check[source.TARGET].astype(bool)))
    objects = pd.read_csv(ROOT / "data/raw/справочник_объектов_диспетчер.csv", dtype=str).set_index("ид_объект")
    cards = []
    for position, (_, row) in enumerate(latest.iterrows()):
        oid = str(row.object_id)
        obj = objects.loc[oid]
        parent_id = str(obj["родитель"])
        parent_name = (str(objects.loc[parent_id, "диспетчерское_название_объекта"])
                       if parent_id in objects.index else f"Группа {parent_id}")
        terms = contributions[position, :-1]
        top = [int(j) for j in np.argsort(terms)[::-1] if terms[j] > 0][:3]
        explanation = [{"feature": cfg["features"][j], "label": LABELS.get(cfg["features"][j], cfg["features"][j]),
                        "value": value_text(x.iloc[position, j]), "contribution_log_odds": float(terms[j])} for j in top]
        observed = int(row.observed_channels)
        action = ("Проверить поступление данных: в день признаков записей от каналов объекта нет. Затем оценить необходимость осмотра."
                  if not observed else
                  "Сопоставить журналы каналов объекта и плановые работы; определить причину сигналов и необходимость осмотра. Решение принимает диспетчер.")
        cards.append({"id": f"object_{feature_date}_{oid}", "entity_mode": "object", "object_id": oid,
            "object_name": str(obj["диспетчерское_название_объекта"]), "object_kind": str(row.object_kind),
            "parent_id": parent_id, "parent_name": parent_name,
            "risk_type": "Зарегистрированная тревога на объекте", "score": float(row.score),
            "warning": bool(row.score >= threshold), "current_alarm": bool(row.alarm_today),
            "events_today": int(row.events_count), "alarm_channels": int(row.alarm_channels),
            "observed_channels": observed, "catalog_channels": int(row.catalog_channels),
            "alarm_days_7d": int(round(float(row.alarm_today_mean7)*7)),
            "observed_days_7d": int(row.observed_days_7d), "days_since_alarm": int(row.days_since_alarm),
            "observation_note": "Нет записей в день признаков; исправность не подтверждена." if not observed else "Есть записи в день признаков; их наличие не доказывает полное покрытие наблюдений.",
            "explanation": explanation, "suggested_action": action,
            "actual_target_alarm": actual[oid]})
    cards.sort(key=lambda item: (-item["score"], item["object_id"]))
    evaluation = pd.read_csv(ROOT / "reports/object_forward_check/metrics.csv")
    row = evaluation[(evaluation.model == "lightgbm_object") & (evaluation.cohort == "all_object_days")].iloc[0]
    exported = {"mode": "retrospective", "entity_mode": "object", **times,
        "model": "LightGBM · сохранённая объектная модель", "model_sha256": cfg["model_sha256"],
        "threshold": threshold, "threshold_selected_on": "октябрь–ноябрь 2025",
        "risk_type": "Хотя бы одно тревожное сообщение по каналам объекта",
        "score_kind": "uncalibrated_model_score", "total_objects": len(cards), "shown_cards": len(cards),
        "warnings_count": sum(c["warning"] for c in cards),
        "observed_objects_today": sum(c["observed_channels"] > 0 for c in cards),
        "test_metrics": {k: float(row[k]) for k in ["average_precision", "precision", "recall", "f1"]},
        "evaluation": {"unit": "объект-день", "rows": int(row["rows"]), "objects": len(cards), "days": 180,
            "issue_start": "2026-01-01", "issue_end": "2026-06-29",
            "note": "Модель и порог зафиксированы до этой проверки. Журналы 2026 ранее использовались в других экспериментах; период не полностью слепой."},
        "limitations": "Прогноз регистрации тревоги, не физической поломки. Score не калиброван как вероятность. Нет записи не означает исправность. Архивная демонстрация.",
        "scheme_note": "Группы по справочнику объектов; схема не отражает реальные координаты или соединения оборудования.",
        "cards": cards}
    destination = Path(destination) if destination else ROOT / "data/app/object_risk_demo.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(exported, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(destination)
    checks = {"feature_history_end": feature_date, "latest_future_labels_absent_before_inference": True,
        "outcomes_loaded_only_after_inference": True, "minimum_lead_hours": 24,
        "frozen_model_checksum_verified": True, "forward_archive_score_max_difference": difference,
        "objects": len(cards), "coverage": coverage, "feature_count": len(cfg["features"]),
        "export_code_sha256": sha256(__file__), "elapsed_seconds": time.monotonic()-started,
        "label_access": "Stored privately in local export and served only by the explicit outcome endpoint"}
    destination.with_name("object_demo_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2))
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return exported


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-date", default=FEATURE_DATE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    export(args.feature_date, args.output)
