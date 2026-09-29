#!/usr/bin/env python3
"""Produce a reviewable report after the frozen comparison has finished."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

OUT = ROOT / "reports/improved_v1"


def markdown_table(df):
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.4f}"
        return str(value)
    lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    lines += ["| " + " | ".join(map(fmt, row)) + " |" for row in df.itertuples(index=False, name=None)]
    return "\n".join(lines)


def main():
    comparison = pd.read_csv(OUT / "comparison.csv")
    candidates = pd.read_csv(OUT / "validation_comparison.csv")
    selection = json.loads((OUT / "selection.json").read_text())
    pred = pd.read_parquet(OUT / "predictions/test.parquet")
    monthly = []
    for month, group in pred.groupby(pred.date.dt.to_period("M")):
        monthly.append({"month": str(month), "rows": len(group), "positives": int(group.target_alarm_next_24h.sum()),
            "baseline_AP": average_precision_score(group.target_alarm_next_24h, group.baseline_score),
            "improved_AP": average_precision_score(group.target_alarm_next_24h, group.score)})
    monthly = pd.DataFrame(monthly)
    monthly.to_csv(OUT / "monthly_test.csv", index=False)
    # Diagnostic cohorts, without changing target or tuning a new threshold.
    cohorts = []
    for name, mask in [("all", np.ones(len(pred), dtype=bool)),
                       ("no_alarm_on_feature_day", pred.alarm_count == 0),
                       ("observed_on_feature_day", pred.events_count > 0)]:
        group = pred.loc[mask]
        if group.target_alarm_next_24h.nunique() < 2:
            continue
        cohorts.append({"cohort": name, "rows": len(group), "positive_rate": group.target_alarm_next_24h.mean(),
            "baseline_AP": average_precision_score(group.target_alarm_next_24h, group.baseline_score),
            "improved_AP": average_precision_score(group.target_alarm_next_24h, group.score)})
    cohorts = pd.DataFrame(cohorts)
    cohorts.to_csv(OUT / "cohort_test.csv", index=False)
    test = comparison[comparison.split == "test"]
    base = test[test.model == "archived_baseline"].iloc[0]
    new = test[test.model != "archived_baseline"].iloc[0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col, label in zip(axes, ["average_precision", "f1", "roc_auc"], ["Average Precision", "F1", "ROC AUC"]):
        bars = ax.bar(["Baseline", "Improved"], [base[col], new[col]], color=["#94a3b8", "#2563eb"])
        ax.bar_label(bars, fmt="%.3f", padding=4)
        ax.set_ylim(0, min(1.1, max(base[col], new[col])*1.25))
        ax.set_title(label); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Same target and test rows; model and thresholds selected on validation")
    fig.tight_layout(); fig.savefig(OUT / "comparison.png", dpi=170); plt.close(fig)
    policies = pd.read_csv(OUT / "operating_points.csv") if (OUT / "operating_points.csv").stat().st_size > 1 else pd.DataFrame()
    columns = ["model", "average_precision", "roc_auc", "precision", "recall", "f1"]
    report = f'''# Улучшение модели: проверенные результаты

Выбранная модель: **{selection['selected_model']}**. Выбор сделан по Average Precision на validation до расчёта её test-метрик.

## Сравнение на одинаковом тесте

{markdown_table(test[columns])}

AP вырос в **{new.average_precision/base.average_precision:.2f} раза**, F1 — в **{new.f1/base.f1:.2f} раза**. Test: {int(new.rows):,} строк, {int(new.positives):,} положительных примеров. Порог каждой модели выбран на validation; основной новый порог — {new.threshold:.8f} по максимуму F1.

## Что проверено

- Все 12 исходных файлов скопированы из Загрузок; SHA256 совпали с оригиналами. Архивы остались в data/raw и исключены из Git.
- Обработаны журналы 2025–2026, как у исходного baseline. Архивы 2019–2024 сохранены для дальнейших исследований, в данной итерации не использованы.
- Сохранённый baseline содержит одно дерево. Воспроизведение его validation-прогнозов: максимальное расхождение {selection['baseline_validation_max_score_difference']:.3g}.
- Train заканчивается по признакам 29.09.2025, validation для выбора — 30.12.2025. Суточные будущие метки не пересекают начало следующей части. Полный старый validation дополнительно показан в comparison.csv для сопоставимости.
- Test использует ту же детерминированную выборку до 1,2 млн строк из января–июня 2026. После test модель не перенастраивалась.
- Проверены временная причинность признаков, изоляция каналов и метрики. Отдельная загрузка сохранённой модели воспроизвела 256 прогнозов; результат в inference_check.json.

## Сравнение кандидатов только на validation

{markdown_table(candidates[['model','iterations','average_precision','precision','recall','f1','seconds']])}

Изменения обучения: выбор итерации по метрике редких событий; обратные веса вероятности отбора вместо двойного балансирования; регуляризация. Дополнительные признаки: давность события, число дней с тревогами/наблюдениями за 3/7/28/90 дней, историческая частота, контекст других каналов объекта и идентификатор канала. В расширенной модели календарные year/month заменены циклическими признаками.

## Устойчивость по месяцам

{markdown_table(monthly)}

## Диагностические подгруппы

{markdown_table(cohorts)}

Подгруппа без тревоги в день признаков проверяет, сохраняется ли сигнал вне продолжения уже активной тревоги. Она не является разметкой физических отказов и не заменяет основной тест.

## Порог высокой точности

{markdown_table(policies[['split','threshold','precision','recall','f1']]) if len(policies) else 'Порог с Precision > 0.7 на validation не найден.'}

Данный дополнительный порог также выбран только на validation. Показатели Precision > 0.7 и Recall > 0.5 оставлены как дополнительный инженерный ориентир. Найденный оригинал ТЗ (§9) не устанавливает эти пороги: показатели определяются при проектировании по качеству данных.

## Ограничения

Цель не менялась: наличие хотя бы одного сообщения о тревоге на следующий календарный день. Физические поломки не подтверждены. Признаки доступны в конце дня; окно будущих 24 часов не означает предупреждение минимум за 24 часа до каждого события.

Для сопоставимости сохранена полная сетка каналов и дат: отсутствие записи остаётся нулём. Данный ноль не доказывает исправность или наблюдаемость. Новые признаки отмечают отсутствие наблюдений, но не исправляют исходную разметку. Требуется отдельная оценка эксплуатационного охвата.

Справочник состояний содержит конфликт для одного ключа и не используется для автоматического изменения целевых меток. Аудит записан в state_dictionary_audit.json.

Test уже присутствовал в старых аналитических отчётах команды, поэтому он не является полностью новым внешним набором. В текущей итерации выбор сделан на validation; следующий убедительный шаг — проверка на новых будущих данных и подтверждённых инцидентах. Score не прошёл отдельную проверку калибровки; не выдавать его за доказанную вероятность физического отказа.

## Воспроизведение

Из корня репозитория, после установки пакетов из requirements-experiments.lock.txt:

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -u src/modeling/improve_baseline.py
.venv/bin/python src/modeling/summarize_improvement.py
.venv/bin/python src/modeling/predict_risk.py --features data/interim/scoring_smoke.parquet --output reports/improved_v1/predictions/scoring_smoke.csv
```

Для нового независимого эксперимента следует сохранить отдельный каталог результатов и зафиксировать новый протокол до просмотра теста. Повтор команды выше воспроизводит текущий эксперимент и перезаписывает его результаты.
'''
    (OUT / "report.md").write_text(report, encoding="utf-8")
    print(report[:1800])


if __name__ == "__main__":
    main()
