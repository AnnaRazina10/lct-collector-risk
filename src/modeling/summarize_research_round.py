"""Summarize separate static, rolling-update, and object-level experiments."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def table(df):
    rows=['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    for row in df.itertuples(index=False,name=None):
        rows.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(rows)

def main():
    out=ROOT/'reports/improved_v3'
    cfg=json.loads((out/'selection.json').read_text())
    results=pd.read_csv(out/'test_comparison.csv')
    tune=pd.read_csv(out/'tuning_candidates.csv')
    monthly=pd.read_csv(out/'test_monthly.csv')
    roll=ROOT/'reports/rolling_update'
    rolling=pd.read_csv(roll/'comparison.csv')
    roll_month=pd.read_csv(roll/'monthly.csv')
    uncertainty=json.loads((out/'uncertainty.json').read_text())
    uncertainty_rows=pd.DataFrame([
        {'model':name,'AP difference 2.5%':interval['lower_2_5'],'AP difference 97.5%':interval['upper_97_5']}
        for name,interval in uncertainty['AP_difference_vs_v2'].items()])
    cols=['model','average_precision','precision','recall','f1','false_positive','true_positive']
    ap_gain=results.iloc[1].average_precision/results.iloc[0].average_precision-1
    if cfg['selected_model']=='v2_control':
        conclusion='Новые кандидаты не превзошли контроль на периоде выбора. Выбран прежний ансамбль; искусственное улучшение не заявляется.'
    elif ap_gain>0:
        conclusion=f'На повторной диагностике AP изменился на {ap_gain:+.1%} относительно v2. Масштаб и компромиссы показаны ниже; сильный результат нельзя обещать по названию метода.'
    else:
        conclusion=f'Выбранный по периоду настройки новый вариант ухудшил AP на диагностике 2026 ({ap_gain:+.1%}). Перевыбор по тесту не выполнялся; v2 сохраняется как предыдущая рабочая версия.'
    report=f'''# Расширенный поиск: измеренные результаты

{conclusion}

## Статическая модель на неизменной цели

Выбран `{cfg['selected_model']}` по AP на октябре–ноябре 2025; структура смеси и число деревьев зафиксированы до дообучения до 29 ноября. Порог {cfg['threshold']:.6f} выбран на декабре. В новом выборе 2026 не участвовал, но этот период ранее уже анализировался. Он не является новым слепым тестом.

{table(results[cols])}

На всех строках одинаковая единица оценки: канал-день, цель — любая зарегистрированная тревога на следующий календарный день. Минимум 24 часа упреждения не гарантирован. Отсутствие записи не является подтверждённой исправностью. Парк каналов и справочник унаследованы из прежней витрины, а не восстановлены по состоянию на каждую дату.

## Все варианты на периоде выбора

Обучены восемь новых компонентов: три LightGBM, три CatBoost, два компонента условного прогноза. Двухэтапный прогноз — произведение вероятности записи и условной вероятности тревоги. Проверены десять заранее заданных смесей. Старый ансамбль до ноябрьского дообучения участвовал как контроль и допустимая составляющая смеси.

{table(tune[['model','average_precision','precision','recall','f1']])}

Метка наличия записи завтра входит только в обучение соответствующей модели и выбор обучающих строк условной модели. При выдаче используется прогноз наблюдаемости, а не будущая запись. Статистики последовательностей обновляются лишь после поступления известного исхода. Пять новых тестов проверяют временную причинность, границы каналов, перенос последнего состояния, завершённые интервалы и выравнивание недельного лага.

## По месяцам: статический вариант

{table(monthly)}

## Отдельная политика ежемесячного обновления

Не является тем же статическим holdout: модель следующего месяца использует известные исходы прошлых месяцев 2026. Число деревьев, признаки и два варианта смешивания заданы до оценки. Вариант `rolling_blend` — 50% обновляемого LightGBM и 50% прежнего CatBoost. Оба порога зафиксированы на декабрьских прогнозах ноябрьских моделей. После оценки веса и пороги не менялись.

{table(rolling[cols])}

Различия относятся ко всей политике обновления и доступности свежих обучающих данных. Подробности — [отчёт ежемесячного обновления](../rolling_update/report.md).

## Устойчивость разницы AP

200 парных повторных выборок семидневных блоков; интервалы 2,5–97,5% разницы AP относительно v2. Выбор модели, порогов и весов по этим интервалам не выполняется. Оценка описательная, условная на уже обученных моделях: переобучение в каждой выборке не проводилось. Для ежемесячных моделей сохраняется преимущество доступа к завершённым прошлым исходам.

{table(uncertainty_rows)}

## Отдельная объектная цель с упреждением 24–48 часов

Ранее обученная модель и порог проверены без изменений на последующих 180 днях 2026 года: 78 объектов, 14 040 объект-дней. Precision 56,91%, Recall 59,58%, AP 0,6213; в среднем 24,44 предупреждения и 10,53 ложных предупреждения на весь парк в день. По сравнению с исторической частотой объектов — на 586 ложных предупреждений меньше при потере 54 верных.

Объектная агрегация меняет частоту положительных исходов, поэтому эти показатели не являются улучшением канальной модели. Цель всё ещё отражает сообщения, а не подтверждённые аварии. Подробности — [объектная проверка](../object_forward_check/report.md).

## Поиск и воспроизведение

[Проверенные источники и применимость](../../docs/deep_search_2026-09-29.md). Дополнительно скачаны 18 файлов исходников; чужие программы не запускались. В коде LEAD обнаружены будущие сдвиги, поэтому его готовые признаки нельзя переносить в наш прогноз без изменения. Исследовательские NHP/S2P2 изучены, но не обучались; TabM был проверен в предыдущей итерации и уступил деревьям.

Требуются исходные данные, окружение requirements-experiments.lock.txt и артефакты v2, воспроизводимые по reports/improved_v2/report.md. Новые команды из корня проекта:

```sh
.venv/bin/python src/modeling/experiment_v3.py prepare
.venv/bin/python src/modeling/experiment_v3.py train
.venv/bin/python src/modeling/experiment_v3.py select
.venv/bin/python src/modeling/experiment_v3.py finalize
.venv/bin/python src/modeling/experiment_v3.py evaluate
.venv/bin/python src/modeling/verify_v3_inference.py
.venv/bin/python src/modeling/rolling_update.py
.venv/bin/python src/modeling/object_forward_check.py
.venv/bin/python src/modeling/uncertainty_research_round.py
.venv/bin/python src/modeling/summarize_research_round.py
.venv/bin/python -m unittest discover -s tests -v
```

Для отдельного прогнозирования подготовленной таблицы признаков используется predict_risk_v3.py. Проверка независимого запуска — inference_cli_check.json. Полные витрины и веса локальны и исключены из Git. Запуски обучения лучше выполнять последовательно из-за расхода памяти; повторные команды перезаписывают артефакты соответствующего эксперимента.
'''
    (out/'report.md').write_text(report)
    roll_text=f'''# Ежемесячное обновление модели

Структура признаков v2, параметры и число деревьев LightGBM сохранены. Перед каждым месяцем используются только завершённые прошлые исходы. Последняя обучающая дата признаков — начало месяца минус два дня; её целевой день строго раньше первого дня нового месяца. Шесть проверенных границ — temporal_checks.json.

На одинаковых 1,2 млн строк 2026:

{table(rolling[cols])}

Два заранее заданных варианта показаны полностью. По результатам 2026 веса и пороги не подбирались. Вариант rolling_blend сохраняет половину вероятности прежнего CatBoost и обновляет вторую половину LightGBM. Пороги рассчитаны до оценки 2026 на декабрьских прогнозах моделей, обученных до ноября.

{table(roll_month[['month','model','average_precision','precision','recall','f1']])}

Проверка представляет последовательную имитацию эксплуатации, а не независимый тест всего 2026: исходы ранних месяцев законно входят в обучение следующих. Использование более свежей информации объясняет часть возможного прироста; сравнение нельзя приписывать только новому алгоритму. Прежние ограничения разметки, наблюдаемости, справочника и next-day горизонта сохраняются.

Протокол — protocol.md, зафиксированные параметры — frozen_before_test.json, границы — temporal_checks.json. Веса всех шести месячных моделей сохранены локально. В реальной эксплуатации требуются такая же доступность завершённых журналов и регламент обновления.
'''
    (roll/'report.md').write_text(roll_text)
    shown=[results.iloc[0].average_precision,results.iloc[1].average_precision,*rolling.iloc[1:].average_precision]
    fig,ax=plt.subplots(figsize=(10,4.6))
    bars=ax.bar(['v2 frozen','v3 selected','Monthly LGB','Monthly blend'],shown,color=['#94a3b8','#148b7d','#3478bf','#7157ad'])
    ax.bar_label(bars,fmt='%.4f',padding=4);ax.set_ylim(0,max(shown)*1.22)
    ax.set_ylabel('Average Precision');ax.set_title('Same channel-day rows; monthly models use completed past updates')
    ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(out/'comparison.png',dpi=170);plt.close(fig)
    print(results[cols].to_string(index=False));print(rolling[cols].to_string(index=False))

if __name__=='__main__':main()
