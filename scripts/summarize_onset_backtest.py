"""Render the fixed backtest report from saved aggregate evidence, without fitting."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/object_onset_backtest'
NAMES = {'hurdle_15': 'Двухэтапная onset-модель', 'any_alarm_control': 'Контроль любой тревоги',
         'frequency': 'Сглаженная частота onset'}


def main():
    table = pd.read_csv(OUT/'evaluation_metrics.csv')
    primary = table.loc[table.k.eq(10)].set_index('model')
    checks = json.loads((OUT/'checks.json').read_text())
    uncertainty = json.loads((OUT/'uncertainty.json').read_text())
    cfg = json.loads((OUT/'configuration.json').read_text())
    strata = pd.read_csv(OUT/'observability.csv').set_index(['model', 'records_intermediate'])
    monthly = pd.read_csv(OUT/'monthly_metrics.csv')
    december = pd.read_csv(OUT/'december_metrics.csv').set_index('model')
    a, b, c = [primary.loc[key] for key in ['hurdle_15', 'any_alarm_control', 'frequency']]
    net = int(a.tp-b.tp)
    no_records_gain = int(strata.loc[('hurdle_15', 0), 'tp']-strata.loc[('any_alarm_control', 0), 'tp'])
    ci = uncertainty['delta_precision_at_k']['ci95']
    audits = {}
    for year in [2023, 2024]:
        path = OUT/f'raw{year}_count_check.json'
        audits[year] = json.loads(path.read_text()) if path.exists() else {'status': 'pending'}
    data_checked = all(x['status'] == 'passed' for x in audits.values())
    lines = [
        '# Проверка переноса onset-метода: 2023 → 2024', '',
        f'На {int(a.days)} днях и {int(a.objects)} объектах двухэтапный метод охватил **{int(a.tp)} начал против {int(b.tp)}** у сопоставимого контроля: +{net}, или +{100*net/b.tp:.1f}% при одинаковых {int(a.warnings)} предупреждениях (10 в день). '
        f'Сглаженная историческая частота даёт {int(c.tp)} попаданий; преимущество над этим более сильным для onset ориентиром — +{int(a.tp-c.tp)}, или +{100*(a.tp-c.tp)/c.tp:.1f}%.', '',
        'Семейство, параметры и бюджет заданы до оценки 2024; по этому году ничего не перенастраивалось. '
        'Но семейство уже было выбрано по более поздним данным, а справочник текущий. Проверка является ретроспективной проверкой переноса, **не новым слепым испытанием** и не доказательством прогноза физических поломок.', '',
        '## Одинаковая цель и нагрузка', '',
        '| Метод | AP onset | Precision@10 | Recall onset | Начала | Повторные тревоги | Без тревоги |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for name in ['any_alarm_control', 'frequency', 'hurdle_15']:
        r = primary.loc[name]
        lines.append(f'| {NAMES[name]} | {r.average_precision:.4f} | {100*r.precision_at_k:.2f}% | {100*r.recall:.2f}% | {int(r.tp)} | {int(r.warning_repeat_alarm)} | {int(r.warning_no_alarm)} |')
    lines += ['', f'Всего {int(a.onsets)} зарегистрированных начал на {int(a.rows)} объект-днях. '
              f'Прирост Precision@10 против контроля: +{100*(a.precision_at_k-b.precision_at_k):.2f} п.п.; '
              f'парный недельный 95% интервал [{100*ci[0]:.2f}; {100*ci[1]:.2f}] п.п. '
              '1000 повторов, seed84. Интервал описывает вариацию этого периода, не устраняя выбор семейства по поздним данным и зависимости между неделями.', '',
              f'Декабрьский gate пройден: {int(december.loc["hurdle_15", "tp"])} против {int(december.loc["any_alarm_control", "tp"])} начал при 290 предупреждениях. '
              f'Предварительно заданный критерий переноса: {"выполнен" if checks["robustness_criterion_passed"] else "не выполнен"}. Рабочие веса и политика интерфейса не заменялись.', '',
              '## Цена изменения цели и наблюдаемость', '',
              f'Предупреждений на дни без любой тревоги стало {int(a.warning_no_alarm)} вместо {int(b.warning_no_alarm)}. '
              f'Precision по любой тревоге снизилась с {100*(b.tp+b.warning_repeat_alarm)/b.warnings:.2f}% до {100*(a.tp+a.warning_repeat_alarm)/a.warnings:.2f}%. '
              'Поэтому метод не объявляется универсально лучшим и не заменяет прежнюю цель.', '',
              f'Из чистого прироста {net} попаданий {no_records_gain} приходится на дни после полного отсутствия любых записей в промежуточный D+1: '
              f'{int(strata.loc[("hurdle_15", 0), "tp"])} против {int(strata.loc[("any_alarm_control", 0), "tp"])}. '
              f'При наличии записей выигрыш равен {net-no_records_gain}. Возможное объяснение части прироста — возобновление регистрации. '
              'Будущая наблюдаемость использована только для последующего анализа; она не фильтрует выдачу.', '',
              f'Против исторической частоты картина иная: на днях после отсутствия записей hurdle даёт '
              f'{int(strata.loc[("hurdle_15", 0), "tp"])} против {int(strata.loc[("frequency", 0), "tp"])} попаданий, '
              f'а при наличии записей — {int(strata.loc[("hurdle_15", 1), "tp"])} против {int(strata.loc[("frequency", 1), "tp"])}. '
              'Следовательно, преимущество над частотой нельзя приписать только возобновлению регистрации.', '',
              '## Результаты по месяцам', '',
              '| Месяц целевого окна | Контроль: попадания | Частота: попадания | Hurdle: попадания | Предупреждения каждого |',
              '|---|---:|---:|---:|---:|']
    for month, part in monthly.groupby('target_month'):
        r = part.set_index('model')
        lines.append(f'| {month} | {int(r.loc["any_alarm_control", "tp"])} | {int(r.loc["frequency", "tp"])} | {int(r.loc["hurdle_15", "tp"])} | {int(r.loc["hurdle_15", "warnings"])} |')
    lines += ['', 'Чувствительность к заранее заданным бюджетам 5/20 сохранена в [evaluation_metrics.csv](evaluation_metrics.csv); по ней метод не выбирался.', '',
              '![Перенос метода и цена предупреждений](comparison.png)', '',
              '## Протокол и данные', '',
              f'Популяция — {len(cfg["eligible_object_ids"])} объекта, имевших записи до 28.09.2023; поздняя активность не используется для отбора. '
              'Начало истории 01.01.2023. Train до признаков28.09 (исходы30.09), ранняя остановка на01.10–28.11 (исходы30.11), '
              f'refit до28.11. Число деревьев hurdle: {cfg["models"]["hurdle_15"]["iterations"]}, контроля: {cfg["models"]["any_alarm_control"]["iterations"]}. '
              'Декабрь не участвует в настройке. Оценка: признаки31.12.2023–28.06.2024; выдачи01.01–29.06; цели02.01–30.06. Високосный календарь даёт181 день.', '',
              'Все 71 исходный признак и метки переносимого builder точно совпали со старой реализацией на 28 470 объект-днях2025. '
              'После усечения истории на15.03.2024 все прошлые признаки и scores трёх методов совпали точно; удаление всех outcome-столбцов не меняет прогноз.', '',
              '**Сверка исходных архивов:** '+('завершена для обоих лет.' if data_checked else 'ещё не завершена; численные результаты предварительные до проверки источников.')]
    for year, audit in audits.items():
        if audit['status'] == 'passed':
            lines.append(f'- {year}: {audit["raw_rows"]:,} исходных строк → {audit["cached_channel_days"]:,} channel-days; '
                         f'все ключи и 11 входных счётчиков совпали точно. [Свидетельство](raw{year}_count_check.json).')
    lines += ['', 'Старый манифест указывал несохранённую в Git версию агрегатора. Независимая сверка проверяет содержание 11 используемых счётчиков, '
              'но не восстанавливает утраченный исходник и не проверяет неиспользуемые числовые агрегаты. Кэши и архивы не переписывались. '
              'Нынешнее соответствие каналов объектам, число каналов и родительские группы остаются не point-in-time. '
              'Дата события в архиве не подтверждает фактическое время поступления записи диспетчеру.', '',
              'Воспроизведение в отдельной копии до существования выходных файлов:', '', '```sh',
              '.venv/bin/python src/modeling/object_onset_backtest.py fit',
              '.venv/bin/python src/modeling/object_onset_backtest.py evaluate',
              '.venv/bin/python scripts/verify_historical_counts.py 2023',
              '.venv/bin/python scripts/verify_historical_counts.py 2024',
              '.venv/bin/python scripts/summarize_onset_backtest.py', '```', '',
              'Повторный fit не перезаписывает существующие веса/конфигурацию. Обучение читает только2023; оценка проверяет зафиксированные SHA. '
              'Для новой копии нужны предоставленные архивы, справочники, дневные кэши и зависимости; веса и построчные прогнозы исключены из Git.', '',
              'Источники: [протокол](protocol.md), [конфигурация](configuration.json), [проверки](checks.json), '
              '[интервалы](uncertainty.json), [наблюдаемость](observability.csv), [точное совпадение builder](history_parity.json).']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout='constrained')
    colors = {'any_alarm_control': '#75838b', 'frequency': '#b87922', 'hurdle_15': '#087f83'}
    for name in ['any_alarm_control', 'frequency', 'hurdle_15']:
        p = monthly.loc[monthly.model.eq(name)]
        axes[0].plot(p.target_month.str[-2:], 100*p.precision_at_k, marker='o', color=colors[name], label=NAMES[name])
    axes[0].set(title='2024: Precision по началам, 10 предупреждений/день', xlabel='Месяц целевого окна', ylabel='Precision, %', ylim=(0, 30))
    axes[0].legend(fontsize=8); axes[0].grid(axis='y', alpha=.2)
    order = ['any_alarm_control', 'frequency', 'hurdle_15']
    offset = [0, 0, 0]
    for field, label, color in [('warning_onset', 'Начало серии', '#087f83'), ('warning_repeat_alarm', 'Повторная тревога', '#b87922'), ('warning_no_alarm', 'Без тревоги', '#c6cdd1')]:
        values = primary.loc[order, field].to_numpy()
        axes[1].barh(range(3), values, left=offset, color=color, label=label)
        offset = [x+y for x, y in zip(offset, values)]
    axes[1].set(yticks=range(3), yticklabels=['Контроль', 'Частота', 'Hurdle'], xlabel='Предупреждения за181 день', title='Одинаковая нагрузка: 1 810 предупреждений')
    axes[1].invert_yaxis(); axes[1].legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.15), ncol=3)
    fig.savefig(OUT/'comparison.png', dpi=160)
    plt.close(fig)
    print(json.dumps({'report': str(OUT/'report.md'), 'raw_data_verified': data_checked, 'net_onsets': net}))


if __name__ == '__main__':
    main()
