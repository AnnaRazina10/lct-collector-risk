"""Produce the onset comparison report from saved aggregate measurements."""
from pathlib import Path
import ast
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/object_onset_experiment'


def pct(value):return f'{value*100:.2f}%'.replace('.',',')


def main():
    metrics=pd.read_csv(OUT/'evaluation_metrics.csv')
    tuning=pd.read_csv(OUT/'tuning_metrics.csv')
    december=pd.read_csv(OUT/'december_metrics.csv').set_index('model')
    monthly=pd.read_csv(OUT/'monthly_metrics.csv')
    strata=pd.read_csv(OUT/'observability.csv')
    cfg=json.loads((OUT/'refit.json').read_text())
    uncertainty=json.loads((OUT/'uncertainty.json').read_text())
    chosen=metrics.loc[metrics.k.eq(10)].set_index('model')
    new=chosen.loc['selected'];old=chosen.loc['any_alarm_refit']
    delta=float(new.precision_at_k-old.precision_at_k);extra=int(new.tp-old.tp)
    interval=uncertainty['delta_precision_at_k']['ci95']
    gate=bool(cfg['december_gate_passed'] and delta>=.02 and interval[0]>0)
    decision={'predeclared_research_gate_passed':gate,'december_gate_passed':cfg['december_gate_passed'],
              'absolute_precision_at_10_gain':delta,'relative_onset_hit_gain':float(new.tp/old.tp-1),
              'extra_onsets_at_same_1800_warnings':extra,'weekly_ci95':interval,
              'production_promoted':False,'new_blind_test_claimed':False,
              'physical_failure_prediction_claimed':False,
              'next_action':'Validate a separately labelled onset queue and the actual notification policy; preserve ongoing-alarm mode.'}
    (OUT/'decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n')
    lines=['# Двухэтапный прогноз начала зарегистрированной серии','',
      f'При одинаковых **1 800 предупреждениях** за 180 дней выбранная модель охватила **{int(new.tp)} начал** против **{int(old.tp)}** у сопоставимого any-alarm контроля: **+{extra} ({pct(new.tp/old.tp-1)} относительно контроля)**. Precision@10 вырос с {pct(old.precision_at_k)} до {pct(new.precision_at_k)}; полнота начал — с {pct(old.recall)} до {pct(new.recall)}. Сравниваются одна и та же onset-цель, 14 040 объект-дней и 2 000 зарегистрированных начал.',
      '', '**Граница вывода:** период 2026 уже исследовался. Результат — сравнительная диагностика на прежних данных, не новая слепая проверка и не доказанное предупреждение физических отказов. Рабочая модель интерфейса не заменена.',
      '', '![Сопоставимое число предупреждений и их исходы](comparison.png)',
      '', '## Постановка и выбор','',
      'Признаки заканчиваются на D; историческая выдача D+1 в 00:00, окно D+2. Положительная метка: на D+1 нет тревожных записей, на D+2 есть. Отсутствие записей не означает исправность. Все 78 объектов и дни сохранены.',
      '', 'Проверены восемь кандидатов: прямой LightGBM, условная двухэтапная модель и четыре совместных состояния будущих двух дней (по две фиксированные конфигурации), регуляризованная логистическая модель, сглаженная частота начал. Дополнительно проверены три заранее заданные смеси лучшего древесного и статистического вариантов. Выбор — только по октябрю–ноябрю 2025, Precision@10, затем AP.',
      '', 'Выбран hurdle_15: P(нет тревоги D+1 | история до D) × P(тревога D+2 | нет тревоги D+1, история до D). Условие второго компонента применяется только к обучающим строкам. При прогнозировании оба компонента используют доступную историю, не фактический D+1. По 188 и 187 деревьев, листья15; веса повторно обучены до признаков28.11.2025. Логистическая модель и смеси уступили по заранее заданному критерию.',
      '', f'Отдельный декабрьский контроль: {int(december.loc["selected","tp"])} начал против {int(december.loc["any_alarm_refit","tp"])} при290предупреждениях, Precision {pct(december.loc["selected","precision_at_k"])} против {pct(december.loc["any_alarm_refit","precision_at_k"])}. Настройки после декабря не изменялись.',
      '', '## Результаты при одинаковой нагрузке','',
      '| Предупреждений в день | Модель | AP по onset | Precision | Полнота начал | Начал охвачено |',
      '|---:|---|---:|---:|---:|---:|']
    for k in [5,10,20]:
        for name,label in [('any_alarm_refit','Any-alarm контроль'),('logistic','Логистическая onset'),('selected','Двухэтапная onset')]:
            row=metrics.loc[metrics.k.eq(k)&metrics.model.eq(name)].iloc[0]
            lines.append(f'| {k} | {label} | {row.average_precision:.4f} | {pct(row.precision_at_k)} | {pct(row.recall)} | {int(row.tp)} |')
    lines += ['', f'Основной бюджет10зафиксирован до расчёта;5и20—чувствительность. Парный недельный bootstrap,1000повторов,27недель (2неполные сохранены): разница Precision@10 **+{delta*100:.2f}п.п.**,95%интервал **[{interval[0]*100:.2f};{interval[1]*100:.2f}]п.п.**. Декабрьский и заранее заданный исследовательский gate пройдены. Bootstrap не устраняет зависимость между неделями, сдвиг данных или повторное использование2026.',
      '', '## Цена изменения цели','', '| Модель | Начало серии | Повторная тревога | Нет тревоги в целевом дне | Всего |','|---|---:|---:|---:|---:|']
    for name,label in [('any_alarm_refit','Any-alarm контроль'),('selected','Двухэтапная onset')]:
        row=chosen.loc[name];p=ast.literal_eval(row.warning_partition)
        lines.append(f'| {label} | {p["onset"]} | {p["repeat_alarm"]} | {p["no_alarm"]} | {int(row.warnings)} |')
    lines += ['', 'Новая модель лучше ранжирует начала, но чаще выбирает дни вообще без тревоги:1 032вместо508. Поэтому нельзя заявлять уменьшение ложных предупреждений по прежней any-alarm цели. Предупреждения на повторную тревогу полезны для отдельной задачи продолжающейся проблемы. Разные очереди требуют разных целей; новая модель не является универсальной заменой прежней.',
      '', '## Наблюдаемость','', '| Записи на D+1 | Начал всего | Контроль: охвачено | Новая модель: охвачено |','|---|---:|---:|---:|']
    for observed,label in [(0,'Нет записей'),(1,'Есть хотя бы одна запись')]:
        part=strata.loc[strata.records_on_intermediate_day.eq(observed)].set_index('model')
        lines.append(f'| {label} | {int(part.loc["selected","onsets"])} | {int(part.loc["any_alarm_refit","true_onset_warnings"])} | {int(part.loc["selected","true_onset_warnings"])} |')
    lines += ['', '75из119дополнительных начал относятся к дню после отсутствия любых записей. Они могут описывать возобновление регистрации, а не новое физическое событие. В страте с записями охват также вырос:325против281. Страты заданы только для последующего анализа; будущая наблюдаемость не использовалась в выдаче. Наличие записей само по себе не доказывает полное покрытие датчиков.',
      '', '## По месяцам, бюджет10','', '| Целевой месяц | Контроль: начал | Новая модель: начал | Предупреждений |','|---|---:|---:|---:|']
    for month,part in monthly.groupby('target_month'):
        part=part.set_index('model')
        lines.append(f'| {month} | {int(part.loc["any_alarm_refit","tp"])} | {int(part.loc["selected","tp"])} | {int(part.loc["selected","warnings"])} |')
    lines += ['', '## Проверки и воспроизведение','',
      'Сохранённые модели загружены заново для оценки. Расчёт с укороченной историей до15марта и без будущих меток воспроизводит более ранние оценки: отклонение0у двухэтапной модели и до1,7e−16у логистической. Последние78оценок воспроизводятся точно без записей29–30июня и без целевых столбцов. Исходы2000начал совпадают с предыдущим аудитом. Контрольные веса, входы, исходники и временные границы записаны в selection/refit/checks.',
      '', '```bash', '.venv/bin/python src/modeling/object_onset_experiment.py select', '.venv/bin/python src/modeling/object_onset_experiment.py refit', '.venv/bin/python src/modeling/object_onset_experiment.py evaluate', '.venv/bin/python scripts/summarize_onset_experiment.py', '```',
      '', 'Фазы select/refit отказываются перезаписывать уже зафиксированную конфигурацию. Для повторного обучения используйте отдельную чистую копию результатов; оценку сохранённой модели можно повторять. Модели и построчные прогнозы локальны, исключены изGit.',
      '', 'Следующий шаг: отдельная ясно названная очередь начала зарегистрированной серии, проверка её реальной политики уведомлений и последующая проверка на новых данных. Нынешняя оценка допускает повторные ежедневные уведомления об объекте; подавление повторов/cooldown требует отдельного последовательного измерения. Потребуются реальные метки ремонтов/подтверждений для утверждений об отказах.',
      '', 'Источники: [узкий обзор первичных методов](../../docs/onset_methods_sources.md). [Протокол](protocol.md), [все кандидаты](tuning_metrics.csv), [декабрь](december_metrics.csv), [итог](evaluation_metrics.csv), [наблюдаемость](observability.csv), [неопределённость](uncertainty.json), [проверки](checks.json).','']
    # Keep numbers and Russian words separated in the reusable report.
    import re
    report='\n'.join(lines)
    report=re.sub(r'(?<=[0-9])(?=[А-Яа-яЁё])|(?<=[А-Яа-яЁё])(?=[0-9])',' ',report)
    for old_text,new_text in [(';5','; 5'),('20—чувствительность','20 — чувствительность'),
                              ('bootstrap,1000','bootstrap, 1 000'),(',27 недель',', 27 недель'),
                              (',95%интервал',', 95% интервал'),(']п.п.','] п.п.'),
                              ('тревоги:1','тревоги: 1'),('вырос:325','вырос: 325'),('изGit','из Git')]:
        report=report.replace(old_text,new_text)
    report += '\nНезависимый аудит: [50 проверок](independent_review.json) и [интерпретация](independent_review.md). Чистый прирост119составляют345попаданий, которых не было у контроля, за вычетом226потерянных. [Полный локальный прогон101теста](integration_tests.log).\n'
    inference_path=OUT/'inference_check.json'
    if inference_path.exists():
        report += '\n[Отдельная проверка inference](inference_check.json): последние 78 оценок совпали точно во всех шести расчётах, без будущих записей и целевых столбцов. Первый расчёт с загрузкой модели и подготовкой всей истории занял 0,497 с (1,826 с с импортами нового Python-процесса), медиана пяти повторов — 0,355 с. Системный кеш не сбрасывался; публикация, сеть и сырые события не измерялись.\n'
    report=re.sub(r'(?<=[0-9])(?=[А-Яа-яЁё])|(?<=[А-Яа-яЁё])(?=[0-9])',' ',report)
    (OUT/'report.md').write_text(report)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={'width_ratios':[1,1.35]})
    labels=['Any-alarm\nконтроль','Двухэтапная\nonset']
    axes[0].bar(labels,[old.tp,new.tp],color=['#647d8a','#0e8d84'],width=.58)
    for i,value in enumerate([old.tp,new.tp]):axes[0].text(i,value+8,str(int(value)),ha='center',fontsize=14,fontweight='bold')
    axes[0].set_ylim(0,510);axes[0].set_ylabel('Охваченные начала из 2 000');axes[0].set_title('Одинаковая нагрузка: 10 в день')
    partitions=[ast.literal_eval(chosen.loc[name,'warning_partition']) for name in ['any_alarm_refit','selected']]
    left=[0,0]
    for key,label,color in [('onset','Начало серии','#0e8d84'),('repeat_alarm','Повторная тревога','#a8beca'),('no_alarm','Нет тревоги','#e3b97b')]:
        values=[p[key] for p in partitions];axes[1].barh(labels,values,left=left,label=label,color=color)
        for i,(start,value) in enumerate(zip(left,values)):axes[1].text(start+value/2,i,str(value),ha='center',va='center',fontsize=11)
        left=[a+b for a,b in zip(left,values)]
    axes[1].set_xlabel('Предупреждения за 180 дней');axes[1].set_title('Исходы всех 1 800 предупреждений');axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3,fontsize=9)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Начало зарегистрированной серии · D+2 · архив 2026',fontsize=15,y=.98)
    fig.tight_layout(rect=[0,.03,1,.94]);fig.savefig(OUT/'comparison.png',dpi=160,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':main()
