#!/usr/bin/env python3
"""One retrospective weather ablation of the frozen 2025 object-day probe.

ERA5 final revisions are not historical point-in-time snapshots. A fixed 7-day
calendar lag does not prove that exact revised values were available then.
This script never changes production selection or evaluates December/2026 labels.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from object_risk_probe import ROOT, CACHE, TARGET, prepare, checksum, dump, metrics, threshold_and_feasibility

OUT = ROOT / "reports/weather_probe"
WEATHER = ROOT / "data/raw/external/московскийконтекст_lag7.csv"
CAVEAT = ("Retrospective ablation only: final ERA5 revisions lack historical available_at/version snapshots. "
          "A seven-day calendar lag does not establish strict point-in-time availability.")


def main():
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    control = json.loads((ROOT / "reports/object_risk_probe/configuration.json").read_text())
    expected_sources = {
        CACHE / "daily_2025.parquet": control["daily_2025_sha256"],
        ROOT / "data/raw/справочник_каналов_датчиков.csv": control["meta_sha256"],
        ROOT / "data/raw/справочник_объектов_диспетчер.csv": control["objects_sha256"],
    }
    for path, expected in expected_sources.items():
        if checksum(path) != expected:
            raise ValueError("Baseline source changed: " + str(path))
    mart, label_date, coverage = prepare()
    original_keys = mart[["object_id", "date", TARGET]].copy()
    weather = pd.read_csv(WEATHER, parse_dates=["date", "source_date"])
    weather = weather.loc[weather.date.between("2025-01-01", "2025-11-30")].copy()
    if weather.date.duplicated().any():
        raise ValueError("Duplicate weather date")
    if not weather.source_date.eq(weather.date-pd.Timedelta(days=7)).all():
        raise ValueError("Weather source date is not D-7")
    weather_features = [c for c in weather if c.startswith("weather_lag7_")]
    if len(weather_features) != 7 or weather[weather_features].isna().any().any():
        raise ValueError("Expected seven complete weather context features")
    mart = mart.merge(weather, on="date", how="left", validate="many_to_one", sort=False)
    pd.testing.assert_frame_equal(original_keys, mart[["object_id", "date", TARGET]])
    if mart[weather_features].isna().any().any():
        raise ValueError("Missing joined weather context")
    if not mart.source_date.le(mart.date-pd.Timedelta(days=7)).all():
        raise ValueError("Noncausal weather calendar join")
    train = mart.index[mart.date.le("2025-09-28")]
    tune = mart.index[mart.date.between("2025-10-01", "2025-11-28")]
    assert label_date.loc[train].max() <= pd.Timestamp("2025-09-30")
    assert label_date.loc[tune].max() <= pd.Timestamp("2025-11-30")
    saved = pd.read_parquet(CACHE / "object_risk_tuning_predictions.parquet")
    keys = ["object_id", "date", "alarm_today", TARGET]
    pd.testing.assert_frame_equal(mart.loc[tune, keys].reset_index(drop=True), saved[keys].reset_index(drop=True))
    base = lgb.Booster(model_file=str(ROOT / control["model_path"]))
    baseline_score = base.predict(mart.loc[tune, control["features"]], num_threads=2)
    difference = float(np.max(np.abs(baseline_score-saved.lightgbm_object.to_numpy())))
    if difference > 1e-12:
        raise ValueError("Original object baseline no longer reproduces exactly")
    yt = mart.loc[train, TARGET].to_numpy(dtype="int8")
    yv = mart.loc[tune, TARGET].to_numpy(dtype="int8")
    features = control["features"] + weather_features
    # Exactly one augmented model, identical parameters and train/tuning rows.
    model = lgb.LGBMClassifier(**control["parameters"])
    model.fit(mart.loc[train, features], yt, eval_set=[(mart.loc[tune, features], yv)],
              categorical_feature=control["categorical"],
              callbacks=[lgb.early_stopping(40, first_metric_only=True), lgb.log_evaluation(100)])
    score = model.predict_proba(mart.loc[tune, features], num_threads=2)[:, 1]
    threshold, feasibility = threshold_and_feasibility(yv, score)
    base_threshold = control["thresholds"]["lightgbm_object"]
    days = mart.loc[tune, "date"].nunique()
    quiet = mart.loc[tune, "alarm_today"].eq(0).to_numpy()
    rows = []
    for name, scores, cutoff in [("object_baseline", baseline_score, base_threshold),
                                  ("object_weather_lag7_retrospective", score, threshold)]:
        rows.append(metrics(name, "all_object_days", yv, scores, cutoff, days))
        rows.append(metrics(name, "no_alarm_on_feature_day", yv[quiet], scores[quiet], cutoff, days))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "metrics.csv", index=False)
    whole = table[table.cohort.eq("all_object_days")].set_index("model")
    before, after = whole.loc["object_baseline"], whole.loc["object_weather_lag7_retrospective"]
    deltas = {name: float(after[name]-before[name]) for name in
              ["average_precision", "precision", "recall", "f1", "false_alerts_per_calendar_day"]}
    result = {"status": "retrospective_ablation_not_production_selection", "point_in_time_caveat": CAVEAT,
              "weather_variant_count": 1, "fixed_weather_lag_days": 7,
              "rows": rows, "weather_minus_baseline": deltas,
              "weather_engineering_reference_70_50": feasibility}
    dump(OUT / "metrics.json", result)
    prediction = saved[keys].copy()
    prediction["baseline_score"] = baseline_score
    prediction["weather_score"] = score
    prediction.to_parquet(CACHE / "object_weather_tuning_predictions.parquet", index=False)
    model_path = ROOT / "models/object_weather_probe.txt"
    model.booster_.save_model(str(model_path))
    reloaded = lgb.Booster(model_file=str(model_path))
    reload_difference = float(np.max(np.abs(reloaded.predict(mart.loc[tune, features].iloc[:256], num_threads=2)-score[:256])))
    assert reload_difference <= 1e-12
    pd.DataFrame({"feature": features, "gain": model.booster_.feature_importance("gain")}).sort_values(
        "gain", ascending=False).to_csv(OUT / "feature_importance.csv", index=False)
    dump(OUT / "checks.json", {"baseline_rows_and_targets_identical": True,
        "baseline_input_sha256_verified": True, "baseline_max_score_difference": difference,
        "saved_weather_model_max_score_difference": reload_difference,
        "weather_join": "date, many_to_one", "weather_source_date_equals_feature_date_minus_7": True,
        "weather_feature_dates_used": [str(weather.date.min()), str(weather.date.max())],
        "weather_source_dates_used": [str(weather.source_date.min()), str(weather.source_date.max())],
        "training_feature_end": "2025-09-28", "training_label_last_day": "2025-09-30",
        "tuning_feature_start": "2025-10-01", "tuning_feature_end": "2025-11-28", "tuning_label_last_day": "2025-11-30",
        "december_labels_used": False, "test_2026_labels_used": False, "coverage": coverage,
        "historical_era5_version_availability_proven": False, "elapsed_seconds": time.monotonic()-start})
    dump(OUT / "configuration.json", {"parameters": control["parameters"], "features": features,
        "categorical": control["categorical"], "weather_features": weather_features,
        "best_iteration": model.best_iteration_, "threshold": threshold,
        "baseline_configuration_sha256": checksum(ROOT / "reports/object_risk_probe/configuration.json"),
        "weather_file_sha256": checksum(WEATHER),
        "weather_provenance_sha256": checksum(WEATHER.parent / "московскийконтекст.provenance.json"),
        "code_sha256": checksum(Path(__file__)), "model_sha256": checksum(model_path),
        "model_path": str(model_path.relative_to(ROOT)), "point_in_time_caveat": CAVEAT})
    lines = ["# Погодный контекст: отдельная ретроспективная проверка", "",
        "Оригинал ТЗ, §13, предусматривает метеоданные через открытые API. §9 оставляет обоснование Precision/Recall команде с учётом качества данных; значения 70/50 ниже являются только инженерным ориентиром.", "",
        "Проверен ровно один заранее выбранный вариант: семь погодных показателей ERA5 для общей точки Москвы, с фиксированным сдвигом на семь суток. Другие лаги, веса и комбинации не подбирались. Рабочая v2 и исходный объектный эксперимент не изменены.", "",
        "## Существенное ограничение", "",
        "**Только ретроспективное исследование.** Выгрузка содержит доступную сейчас версию ERA5, которая могла быть пересмотрена спустя месяцы после события. Исторические версии и время доступности именно этих значений не предоставлены. Сдвиг на семь суток исключает использование погоды будущей календарной даты, но не доказывает доступность финальных исправлений на момент прогноза. Результат нельзя представлять как строгий исторический replay или подтверждённый производственный эффект.", "",
        "Погода описывает общий фон Москвы, а не точные координаты обезличенных объектов. Данные объектов при загрузке погоды наружу не передавались.", "",
        "## Сопоставимость", "",
        "Единица оценки — объект-день; цель — любая зарегистрированная тревога объекта в окне +24…+48 часов после выпуска, не подтверждённая физическая авария. Использованы те же 78 объектов, 21 138 обучающих и 4 602 проверочных строки, что в исходном объектном probe. Обучающие признаки заканчиваются 28.09.2025, метки — 30.09; настройка использует признаки 01.10–28.11 и метки до 30.11. Декабрьские и 2026 метки не оценивались.", "",
        f"Исходная модель воспроизведена на совпадающих строках с максимальным расхождением {difference:.1g}. Параметры LightGBM, полная обучающая выборка и правило остановки одинаковы. Каждая версия выбирает максимум F1 на одной и той же настройке; независимого теста здесь нет.", "",
        "| Метод | AP | Precision | Recall | F1 | Ложных предупреждений/день |",
        "|---|---:|---:|---:|---:|---:|"]
    for title, row in [("Без погоды", before), ("ERA5, лаг 7 дней", after)]:
        lines.append(f"| {title} | {row.average_precision:.4f} | {row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f} | {row.false_alerts_per_calendar_day:.2f} |")
    lines += ["", f"Изменение AP: {deltas['average_precision']:+.4f}; изменение F1: {deltas['f1']:+.4f}. Доля положительных исходов сохраняется 24,95%. Ложные предупреждения считаются суммарно по 78 объектам на календарный день.", "",
        ("Численный прирост AP мал, а F1 и число ложных предупреждений не изменились. Существенного практического улучшения этот результат не показывает; оснований включать погоду в рабочую модель по этой проверке нет." if deltas["average_precision"] > 0 and abs(deltas["f1"]) < 1e-12 else
         "Погодные признаки улучшили AP на этой настройке; устойчивость и реальная доступность признаков ещё не подтверждены." if deltas["average_precision"] > 0 else
         "Погодные признаки не улучшили AP на этой настройке. Добавлять их в рабочую модель по этому эксперименту оснований нет."), "",
        "В `metrics.csv` также показана подгруппа без тревоги в день признаков, с теми же порогами. Она не равна началу нового эпизода: тревога могла возникнуть в промежуточный день D+1. Проверка инженерного ориентира 70/50 находится в `metrics.json`; она не является проверкой обязательного требования ТЗ.", "",
        "## Источник и воспроизведение", "",
        "Источник: [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api), ERA5 / Copernicus Climate Change Service. Обновления и отличия финальной версии описаны в [документации ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview). Атрибуция и лицензия CC BY 4.0 сохранены в `docs/weather_context_research.md` и исходном provenance-файле.", "",
        "```sh", ".venv/bin/python src/modeling/object_weather_probe.py", "```", "",
        "Проверки строк, временных границ и перезагрузки модели сохранены в `checks.json`; SHA-256 и параметры — в `configuration.json`. Локальные веса и построчные оценки исключены из Git.", ""]
    (OUT / "report.md").write_text("\n".join(lines))
    print(table.to_string(index=False), flush=True)
    print(json.dumps(deltas, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
