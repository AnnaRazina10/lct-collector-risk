# LightGBM baseline

Baseline построен по выводам EDA: дневная витрина по каналу и target `target_alarm_next_24h`,
то есть наличие хотя бы одного тревожного события по этому каналу в следующие 24 часа.

## Постановка

- Период: 2025-2026 годы.
- Train: до 2025-09-30 включительно.
- Validation: 2025-10-01 - 2025-12-31.
- Test: с 2026-01-01 до последней доступной даты минус один день.
- Сущность: `ид_канала_данных`.
- Шаг времени: 1 день.
- Всего строк витрины: 6,265,865.
- Строк в обучающей подвыборке LightGBM: 700,000.
- Лучший порог по F1 на validation: 0.0849.

## Метрики

| split | rows | positive_rate | roc_auc | pr_auc | threshold | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 1200000 | 0.01555 | 0.701687 | 0.063291 | 0.0849116 | 0.132376 | 0.172401 | 0.14976 |
| train | 1200000 | 0.00781333 | 0.936263 | 0.09961 | 0.0849116 | 0.141815 | 0.3109 | 0.194781 |
| valid | 1057724 | 0.014397 | 0.747677 | 0.0697783 | 0.0849116 | 0.151696 | 0.212372 | 0.176978 |

## Топ суммарной нормированной важности

| feature | total_normalized_importance |
| --- | --- |
| sensor_type | 1.4225 |
| alarm_count_sum_14d | 0.804147 |
| object_id | 0.654513 |
| dayofweek | 0.293339 |
| month | 0.255639 |
| text_share_1d | 0.0906613 |
| alarm_rate_14d | 0.0841646 |
| norm_count | 0.0731673 |
| undefined_count_sum_14d | 0.0572347 |
| events_count_sum_7d | 0.05054 |
| text_count | 0.0450659 |
| value_min | 0.0426068 |
| numeric_share_14d | 0.0230975 |
| events_count_sum_14d | 0.0224407 |
| numeric_count_sum_14d | 0.0209467 |

## Максимальный drift train vs test

| feature | psi_train_test | ks_train_test |
| --- | --- | --- |
| month | 4.0871 | 0.339158 |
| undefined_count_sum_14d | 0.161798 | 0.157417 |
| text_count_sum_14d | 0.0756136 | 0.125408 |
| events_count_sum_14d | 0.0659489 | 0.119408 |
| value_mean_1d | 0.0509571 | 0.0654445 |
| value_max | 0.0496101 | 0.0637457 |
| text_count_sum_7d | 0.043501 | 0.0974833 |
| value_min | 0.0399389 | 0.0582907 |
| events_count_sum_7d | 0.0359046 | 0.0915834 |
| events_count_sum_3d | 0.0128737 | 0.0644167 |
| value_std_1d | 0.0119553 | 0.0294043 |
| value_range_1d | 0.00511253 | 0.0222113 |
| text_count_sum_3d | 0.00230258 | 0.069175 |
| events_count | 0.00050987 | 0.0292667 |
| dayofweek | 7.75835e-05 | 0.003525 |

## Артефакты

- `figures/roc_auc_curve.png` - ROC-AUC.
- `figures/pr_curve.png` - PR-кривая.
- `figures/target_distribution_by_split.png` - распределение target=0/1.
- `figures/score_distribution_by_target.png` - распределение скоринга по target.
- `figures/threshold_precision_recall_f1.png` - Precision, Recall, F1 по порогам.
- `figures/feature_importance_split.png` - важность LightGBM split.
- `figures/feature_importance_gain.png` - важность LightGBM gain; это стандартная метрика, соответствующая `grain` из запроса.
- `figures/feature_importance_grain_gain_alias.png` - тот же gain-график с alias-именем.
- `figures/feature_importance_shap.png` - SHAP-важность.
- `figures/feature_importance_permutation.png` - permutation importance.
- `figures/feature_importance_total_normalized.png` - суммарная нормированная важность.
- `figures/drift_psi_train_test.png` - PSI train vs test.
- `figures/drift_ks_train_test.png` - KS train vs test.
- `figures/feature_coverage_heatmap.png` - покрытие признаков.
- `tables/feature_drift.parquet` - PSI/KS drift по признакам.
- `tables/feature_coverage.parquet` - coverage признаков.

## Ограничения

Target является proxy-разметкой по техническому признаку `тревожное`, а не подтвержденной аварией.
Baseline нужен как первая воспроизводимая точка отсчета, а не финальная бизнес-модель.
