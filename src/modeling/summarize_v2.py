"""Report frozen temporal experiment without selecting again on test."""
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'reports/improved_v2'

def table(df):
    cols=list(df.columns);rows=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for row in df.itertuples(index=False,name=None):
        rows.append('| '+' | '.join(f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |')
    return '\n'.join(rows)

def main():
    cfg=json.loads((OUT/'selection.json').read_text())
    results=pd.read_csv(OUT/'test_comparison.csv')
    tune=pd.read_csv(OUT/'tuning_candidates.csv').sort_values('average_precision',ascending=False)
    cal=json.loads((OUT/'calibration_metrics.json').read_text())
    monthly=pd.read_csv(OUT/'test_monthly.csv')
    pred=pd.read_parquet(OUT/'predictions/test.parquet')
    cols=['model','average_precision','precision','recall','f1','roc_auc']
    old=pd.read_csv(ROOT/'reports/improved_v1/comparison.csv')
    baseline=old[(old.model=='archived_baseline')&(old.split=='test')]
    display=pd.concat([baseline[cols],results[cols]],ignore_index=True)
    cohorts=[]
    for name,part in [('Без тревоги в день признаков',pred[pred.alarm_count==0]),('Есть записи в день признаков',pred[pred.events_count>0])]:
        cohorts.append({'группа':name,'строк':len(part),'v1_AP':average_precision_score(part.target_alarm_next_24h,part.score_v1),'v2_AP':average_precision_score(part.target_alarm_next_24h,part.score_v2)})
    fig,axs=plt.subplots(1,3,figsize=(13,4.4))
    for ax,metric,title in zip(axs,['average_precision','precision','recall'],['Average Precision','Precision','Recall']):
        bars=ax.bar(['Baseline','v1','v2'],display[metric],color=['#94a3b8','#388cba','#148b7d'])
        ax.bar_label(bars,fmt='%.3f',padding=3);ax.set_title(title);ax.set_ylim(0,max(display[metric])*1.25);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Same next-day alarm target · 1,200,000 test rows');fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)
    ap_gain=results.iloc[1].average_precision/results.iloc[0].average_precision-1
    text=f'''# Вторая итерация: временная динамика и комбинации моделей

Выбран **{cfg['selected_model']}**, критерий — Average Precision на октябре–ноябре 2025. Параметры выбранных компонентов зафиксированы; затем они дообучены до 29 ноября без проверки на декабре. Порог {cfg['threshold']:.6f} выбран отдельно на декабре. Test 2026 не участвовал в новом выборе, но ранее уже просматривался: он не является новым слепым тестом.

## Сопоставимый тест

{table(display)}

AP вырос на {ap_gain:.1%} относительно v1. Полнота увеличилась, но точность предупреждений снизилась: рабочий порог следует согласовывать с допустимой нагрузкой диспетчера.

Сравнение использует те же 1,2 млн строк и прежнюю цель. Итоговый конвейер включает более свежие обучающие данные; прирост нельзя приписывать исключительно новой архитектуре или признакам. Оригинал ТЗ (§9) не задаёт фиксированные Precision 70% / Recall 50%: их определяют на этапе проектирования. Эти числа оставлены в дополнительных расчетах как инженерный ориентир, а не требование организатора. Новая модель оценивает сообщение о тревоге, не подтверждённую физическую поломку.

## Кандидаты до финального дообучения и теста

{table(tune[['model','average_precision','precision','recall','f1']])}

## Отдельный период порога

Декабрь 2025: AP={cal['average_precision']:.4f}, Precision={cal['precision']:.4f}, Recall={cal['recall']:.4f}, F1={cal['f1']:.4f}. Максимальная полнота при Precision>0.7 на данном периоде: {cal['max_recall_at_precision_gt_0_7']:.4f}. Инженерные ориентиры 0.7/0.5 совместно достижимы на этом периоде: {cal['requirement_achievable']}.

## По месяцам и подгруппам

{table(monthly)}

{table(pd.DataFrame(cohorts))}

## Обогащение

Использованы причинные лаги, серии и переходы состояний, экспоненциальная история, отклонения от прошлых числовых показаний, контекст объекта/типа и родительского объекта. При включении внутридневных признаков — только события текущего дня, последние 6/12 часов, первое/последнее состояние. Исторические профили сформированы исключительно по данным до 2025. Состав зафиксирован в selection.json, происхождение истории в history_summary.json.

Внешние датасеты в основную модель не добавлялись. Оригинал ТЗ (§13) предусматривает метеоданные. Скачаны 577 дней и семь погодных показателей Москвы. Отдельная объектная проверка с лагом 7 дней дала AP 0,6266 → 0,6271, без изменения F1 и числа ложных предупреждений. Существенная польза не подтверждена; окончательный реанализ также не доказывает историческую доступность значений. Подробности — reports/weather_probe/report.md. Изученные современные методы и условия применения описаны в docs/sota_enrichment_2026-09-28.md. Не каждый изученный метод запускался: таблица выше перечисляет фактически проверенные варианты.

## Границы результата

- Состав парка унаследован из полной сетки 2025–2026 и текущего справочника: он не реконструирован на каждую историческую дату. Нулевая запись на полной сетке не доказывает исправность. Наблюдаемость представлена признаками, исходная сопоставимая разметка сохраняется.
- При агрегировании суточных данных признаки доступны в конце дня. Прогноз следующего дня не гарантирует минимум 24 часа до каждого события.
- «Обесточен» часто означает состояние фазы без тревоги. Подмена цели техническимproxy может создать красивую, но нерелевантную метрику. Аудит — docs/target_audit_v2.md.
- В первоначальной копии требований были ошибочно записаны фиксированные пороги. Оригинал docs/materials/8. ДЖКХ.pdf задаёт выбор показателей при проектировании; по-прежнему требуются содержательная цель, корректный момент прогноза и будущая эксплуатационная проверка.
- Для v1 декабрь уже использовался при выборе: декабрь второй итерации — отдельный от нового выбора период порога, но не никогда не виденные данные.

## Воспроизведение

Протокол — protocol.md. Команды из корня проекта после установки зависимостей и подготовки v1 (исходная витрина, метаданные и контрольные прогнозы используются повторно):

Требуется Python 3.12 и окружение с requirements-experiments.lock.txt (установка: `uv pip install --python .venv/bin/python -r requirements-experiments.lock.txt`). Исходные 12 файлов должны находиться в data/raw. Сохранённые артефакты baseline находятся в reports/baseline_lightgbm и уже входят в репозиторий; при их отсутствии сначала выполните `src/modeling/lightgbm_baseline.py`. Следующие первые две команды заново создают исключённые из Git результаты v1, необходимые для контроля и сравнения.

```sh
.venv/bin/python src/modeling/improve_baseline.py
.venv/bin/python src/modeling/summarize_improvement.py
.venv/bin/python src/modeling/cache_older_history.py --fail-fast
.venv/bin/python src/modeling/intraday_features.py
.venv/bin/python src/modeling/experiment_v2.py prepare
.venv/bin/python src/modeling/experiment_v2.py freeze
.venv/bin/python src/modeling/experiment_v2.py train --prior data/interim/channel_prior_frozen_v2.parquet --intraday
.venv/bin/python src/modeling/tabm_candidate.py --device cpu --seconds 360 --threads 3
.venv/bin/python src/modeling/experiment_v2.py select --choose-only
.venv/bin/python src/modeling/experiment_v2.py finalize
.venv/bin/python src/modeling/experiment_v2.py evaluate
.venv/bin/python src/modeling/summarize_v2.py
```

Дополнительный TabM запускается отдельным tabm_candidate.py до этапа select; параметры и время — tabm_metrics.json. Повтор перезаписывает результаты данной итерации. Для дальнейшего улучшения необходимо новое имя эксперимента и заранее заданный протокол.
'''
    (OUT/'report.md').write_text(text)
    print(display.to_string(index=False))

if __name__=='__main__':main()
