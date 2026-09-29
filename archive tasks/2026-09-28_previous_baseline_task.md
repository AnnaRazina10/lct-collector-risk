# Current Task: LightGBM baseline

## Статус

EDA выполнен и интерпретирован. Текущая задача - построить первый воспроизводимый LightGBM baseline на основе выводов `reports/eda/eda_interpretation.md`.

## Задача

Нужно построить дневную витрину по каналам датчиков, обучить LightGBM для proxy-target `target_alarm_next_24h`, оценить качество на временном train/validation/test split и подготовить графики, таблицы, parquet-артефакты drift/coverage и отчет.

## Что нужно получить

Нужно получить ROC-AUC и PR-кривые, распределение target=0/1 по train/validation/test, график Precision, Recall и F1 в зависимости от порога, важности признаков LightGBM split/gain, SHAP и permutation, суммарную нормированную важность, parquet-файлы feature drift и feature coverage, графики PSI, KS и coverage.

## Основные направления анализа

Нужно проверить, насколько proxy-target предсказуем на дневной витрине, какие признаки дают вклад в модель, насколько различаются train/validation/test по покрытию и распределениям признаков, а также какие признаки имеют заметный drift между обучающим и тестовым периодом.

## Критерии готовности

Задача считается выполненной, когда есть воспроизводимый скрипт baseline, обученная модель, markdown-отчет, метрики, графики, таблицы важностей, parquet-файлы drift/coverage и понятные ограничения выбранного proxy-target.

## Где должны появиться результаты

Результаты baseline должны быть сохранены в `reports/baseline_lightgbm/`, код - в `src/modeling/`, а новые директории должны быть отражены в `inventory.md`.
