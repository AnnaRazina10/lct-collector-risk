"""Audit fixed object forecasts and summarize method and operational tradeoffs."""
import json
import numpy as np
import pandas as pd
import object_risk_probe as base
import object_combined as combined
from uncertainty_research_round import WeightedAP


def table(frame):
    text = ['| '+' | '.join(frame.columns)+' |', '| '+' | '.join(['---']*len(frame.columns))+' |']
    for row in frame.itertuples(index=False, name=None):
        text.append('| '+' | '.join(f'{x:.4f}' if isinstance(x, (float, np.floating)) else str(x) for x in row)+' |')
    return '\n'.join(text)


def main():
    out = combined.OUT
    cfg = json.loads((out/'selection.json').read_text())
    pred = pd.read_parquet(base.CACHE/'object_combined_test.parquet')
    # No June29/30 records and no target column are available to this reload.
    prefix, _, _ = base.prepare(end_date='2026-06-28')
    prefix = prefix.drop(columns=[base.TARGET])
    scores = combined.component_scores(prefix, cfg)
    prefix['selected'] = scores['selected']
    last = prefix.loc[prefix.date.eq('2026-06-28'), ['object_id', 'date', 'selected']]
    checked = last.merge(pred[['object_id', 'date', 'selected']], on=['object_id', 'date'], validate='one_to_one')
    assert len(checked) == 78
    error = float(np.max(np.abs(checked.selected_x-checked.selected_y)))
    if error > 1e-12:
        raise ValueError('Truncating future changed the last forecast')
    days, unique = pd.factorize(pred.date, sort=True)
    metrics = {name: WeightedAP(pred[base.TARGET], pred[name], days) for name in ['selected', 'lightgbm_control', 'previous_frozen']}
    rng = np.random.default_rng(139); draws = []
    for _ in range(200):
        starts = rng.integers(0, len(unique), int(np.ceil(len(unique)/7)))
        sampled = ((starts[:, None]+np.arange(7)) % len(unique)).ravel()[:len(unique)]
        weights = np.bincount(sampled, minlength=len(unique))
        draws.append(metrics['selected'](weights)-metrics['lightgbm_control'](weights))
    bounds = np.quantile(draws, [.025, .975])
    base.dump(out/'audit.json', {'prefix_rows': 78, 'future_records_available': False, 'target_in_inference_frame': False,
        'maximum_difference': error, 'selection_sha256': base.checksum(out/'selection.json'),
        'AP_difference_vs_refit_control_interval': bounds.tolist(), 'bootstrap_draws': 200, 'block_days': 7, 'seed': 139,
        'uncertainty_note': 'Descriptive, conditional on fixed fitted models; no model reselection'})
    comparison = pd.read_csv(out/'test_metrics.csv')
    budgets = pd.read_csv(out/'daily_budget.csv')
    tuning = pd.read_csv(out/'tuning_mixtures.csv')
    statistical = pd.read_csv(combined.STAT/'tuning.csv')
    sequence = pd.read_csv(combined.SEQUENCE/'tuning_metrics.csv')
    cols = ['model', 'average_precision', 'precision', 'recall', 'f1', 'true_positive', 'false_positive', 'all_alerts_per_calendar_day']
    text = f'''# Временные и статистические модели: результат комплексного сравнения

Сильного превосходства над бустингом не получено. Компактная GRU уступила контролю. Лучшая по периоду настройки смесь — 75% LightGBM и 25% сглаженной сезонной статистики — немного улучшила AP, но на выбранном заранее декабрьском пороге снизила полноту и F1. При одинаковом бюджете проверок практическая разница с контролем мала. Автоматическая замена модели демонстрации по этому результату не выполняется.

## Сопоставимая постановка

78 объектов; 14 040 объект-дней; выпуск прогнозов 01.01–29.06.2026. История заканчивается в день D; выдача D+1 00:00; целевой интервал [D+2 00:00,D+3 00:00), то есть 24–48 часов упреждения. Цель — любая зарегистрированная тревога, а не подтверждённый физический отказ. Принадлежность каналов объектам задана текущим справочником и не восстановлена на каждую историческую дату. Отсутствие записей не доказывает исправность оборудования.

Одиночные кандидаты обучались до сентября 2025, выбор сделан по октябрю–ноябрю. После выбора выбранная смесь и контроль дообучены до 28 ноября с исходами до 30 ноября. Пороги определены на 1–29 декабря (целевые дни до31 декабря), до повторной проверки 2026. Период 2026 ранее уже анализировался; результат не является новым слепым тестом.

## Проверка 2026

`selected` — выбранная смесь; `lightgbm_control` — сопоставимо дообученный контроль; `previous_frozen` — прежняя модель объектной демонстрации. Исходные прогнозы старого контроля точно воспроизведены.

{table(comparison.loc[comparison.cohort.eq('all_object_days'), cols])}

Снижение ложных предупреждений у смеси сопровождается пропуском большего числа тревог: относительно сопоставимого контроля на 495 ложных предупреждений меньше, но и на 296 верных предупреждений меньше. Утверждать безусловное снижение эксплуатационных затрат без цены пропуска/проверки нельзя.

95%-описательный интервал разницы AP смеси и сопоставимого контроля по 200 парным выборкам недельных блоков: [{bounds[0]:.5f}; {bounds[1]:.5f}]. Модели в bootstrap не переобучались; неопределённость условна на текущих весах. По интервалу модель не перевыбиралась.

## Одинаковая ежедневная нагрузка

Бюджеты 5/10/20 объектов в день были заданы до теста. Порядок при совпадении score определяется object_id. Ни один бюджет не объявляется оптимальным без стоимости и возможностей диспетчерской службы.

{table(budgets[['model', 'objects_per_day', 'precision', 'recall', 'true_positive', 'false_positive']])}

При 10 объектах в день сопоставимый LightGBM даёт Precision 71,78%, Recall 30,74%; смесь — 71,72% и 30,72%. Высокая точность достигается ценой ограниченной полноты, а не выполнением несуществующего обязательного порога 70/50.

## Объекты без тревоги в день признаков

{table(comparison.loc[comparison.cohort.eq('no_alarm_on_feature_day'), cols])}

Данный cohort не исключает начала тревоги в промежуточный день D+1 и не равен строгому прогнозу начала нового эпизода. Для доказанного прогнозирования новых отказов нужны согласованная событийная разметка, интервалы наблюдения и подтверждения.

## Разные семейства на периоде выбора

{table(statistical[['model', 'average_precision', 'precision', 'recall', 'f1']])}

{table(sequence.loc[sequence.cohort.eq('all_object_days') & sequence.model.str.startswith('gru_'), ['model', 'average_precision', 'precision', 'recall', 'f1']])}

{table(tuning)}

Марковская модель имеет три состояния: отсутствие записи, запись без тревоги, тревога; её переходы обновляются только по известным событиям, прогноз делается на два шага. Сезонная модель использует сглаженные частоты объекта по целевому дню недели с полураспадом 90 дней. Логистическая авторегрессия и сплайны обучают преобразования только на train. GRU содержит 4577 параметров, окна28/56 дней, маску наличия истории; два запуска реально выполнены и сохранены, а не обозначены названием архитектуры.

## Проверки и воспроизведение

Из последнего повторного запуска удалены записи 29–30 июня и целевой столбец. Все78 прогнозов на признаках28 июня совпали с ранее сохранёнными (maxdiff={error:.3g}). Проверены причинность статистик, двухшаговый переход, целевой день недели, train-only scaler нейросети, границы окон и воспроизведение загруженных моделей.

```sh
.venv/bin/python src/modeling/object_statistical_candidate.py
.venv/bin/python src/modeling/object_sequence_candidate.py
.venv/bin/python src/modeling/object_combined.py select
.venv/bin/python src/modeling/object_combined.py finalize
.venv/bin/python src/modeling/object_combined.py evaluate
.venv/bin/python src/modeling/summarize_object_candidates.py
```

Нужны локальные daily2025/2026, справочники и прежний object_risk_probe. Команды перезаписывают артефакты своего эксперимента. Политика до оценки — protocol.md; выбор — selection_before_refit.json; дообучение до порога — refit_before_calibration.json; итог — selection.json. Веса и построчные прогнозы локальны, исключены из Git. Промышленная готовность и победа этим сравнением не доказаны.
'''
    (out/'report.md').write_text(text)
    (combined.STAT/'report.md').write_text('# Статистические альтернативы\n\nОбучение/выбор ограничены январём–ноябрём2025.\n\n'+table(statistical[['model','average_precision','precision','recall','f1']])+'\n\nПолное сопоставимое продолжение, ограничения и команды — [совместный отчёт](../object_combined/report.md).\n')
    print('Audited last 78 forecasts without future data; AP difference interval:', bounds)


if __name__ == '__main__':
    main()
