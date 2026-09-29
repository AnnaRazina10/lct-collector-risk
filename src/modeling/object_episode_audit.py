"""Post-selection audit of registered object alarm starts, with fixed forecasts."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pandas as pd

import object_risk_probe as base
import object_combined as combined

ROOT = base.ROOT
OUT = ROOT / 'reports/object_episode_audit'
PREDICTION = base.CACHE / 'object_combined_test.parquet'
GAPS = (1, 3, 7)
MODELS = ('previous_frozen', 'lightgbm_control', 'selected')
START = pd.Timestamp('2025-12-31')
END = pd.Timestamp('2026-06-28')
LABEL_END = pd.Timestamp('2026-06-30')


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def episode_labels(history, gap):
    """Gap is g exact calendar days before T, excluding T itself.

    Missing history remains unknown. An explicitly present zero-event day means
    no registered records, never certified healthy equipment.
    """
    if gap < 1:
        raise ValueError('Gap must be positive')
    columns = ['object_id', 'date', 'alarm_today', 'observed_today']
    history = history[columns].copy()
    if history[['object_id', 'date']].duplicated().any():
        raise ValueError('Duplicate object/day')
    if history[columns[2:]].isna().any().any() or not history[columns[2:]].isin([0, 1]).all().all():
        raise ValueError('Alarm and observation must be explicit binary values')
    if (history.alarm_today > history.observed_today).any():
        raise ValueError('Registered alarm requires at least one registered record')
    parts = []
    for object_id, group in history.groupby('object_id', observed=True, sort=False):
        daily = group.sort_values('date').set_index('date')
        calendar = pd.date_range(daily.index.min(), daily.index.max())
        calendar_values = daily.reindex(calendar)
        past_alarm = calendar_values.alarm_today.shift(1).rolling(gap, min_periods=gap).sum()
        past_observed = calendar_values.observed_today.shift(1).rolling(gap, min_periods=gap).sum()
        past_days = calendar_values.alarm_today.shift(1).rolling(gap, min_periods=1).count()
        rows = daily.copy()
        rows['context_days'] = past_days.reindex(rows.index).fillna(0).astype(int)
        rows['context_known'] = rows.context_days.eq(gap)
        rows['gap_alarm_days'] = past_alarm.reindex(rows.index)
        rows['gap_record_days'] = past_observed.reindex(rows.index)
        onset = rows.alarm_today.eq(1) & rows.gap_alarm_days.eq(0)
        rows['episode_start'] = onset.astype('float64').where(rows.context_known)
        rows['repeat_alarm'] = (rows.alarm_today.eq(1) & ~onset).astype('float64').where(rows.context_known)
        rows['gap_record_coverage'] = np.select([
            ~rows.context_known, rows.gap_record_days.eq(gap), rows.gap_record_days.eq(0)],
            ['unknown_context', 'records_every_gap_day', 'no_records_in_gap'], default='records_some_gap_days')
        rows['gap'] = gap
        parts.append(rows.reset_index())
    return pd.concat(parts, ignore_index=True)


def join_forecasts(predictions, labels):
    """Exact target-date join, never a positional shift of prediction rows."""
    result = predictions.rename(columns={'date': 'feature_date'}).copy()
    if result[['object_id', 'feature_date']].duplicated().any():
        raise ValueError('Duplicate forecast identity')
    result['target_date'] = result.feature_date + pd.Timedelta(days=2)
    result['issue_date'] = result.feature_date + pd.Timedelta(days=1)
    right = labels.rename(columns={'date': 'target_date', 'alarm_today': 'target_alarm',
                                   'observed_today': 'target_has_records'})
    result = result.merge(right, on=['object_id', 'target_date'], how='left', validate='one_to_one', indicator=True)
    if not result._merge.eq('both').all():
        raise ValueError('Every exact D+2 target date needs an explicit label row')
    result = result.drop(columns='_merge')
    if not result[base.TARGET].eq(result.target_alarm).all():
        raise ValueError('Saved target differs from target-day alarm')
    return result


def ratio(numerator, denominator):
    return float(numerator/denominator) if denominator else None


def summarize(part, model, threshold):
    warning = part[model].ge(threshold)
    known = part.context_known.astype(bool)
    alarm = part.target_alarm.eq(1)
    onset = part.episode_start.eq(1)
    repeat = part.repeat_alarm.eq(1)
    hit = warning & onset
    repeated_warning = warning & repeat
    no_alarm_warning = warning & known & ~alarm
    unknown_warning = warning & ~known
    assert int(warning.sum()) == int(hit.sum()+repeated_warning.sum()+no_alarm_warning.sum()+unknown_warning.sum())
    return {'model': model, 'threshold': threshold, 'rows': len(part),
            'target_calendar_days': int(part.target_date.nunique()),
            'known_context_rows': int(known.sum()), 'unknown_context_rows': int((~known).sum()),
            'alarm_days': int(alarm.sum()), 'registered_episode_starts': int(onset.sum()),
            'repeated_alarm_days': int(repeat.sum()), 'warned_episode_starts': int(hit.sum()),
            'missed_episode_starts': int((onset & ~warning).sum()),
            'episode_start_recall': ratio(hit.sum(), onset.sum()),
            'warnings_total': int(warning.sum()), 'warnings_on_repeated_alarm_days': int(repeated_warning.sum()),
            'warnings_on_no_alarm_days': int(no_alarm_warning.sum()),
            'warnings_with_unknown_context': int(unknown_warning.sum()),
            'warnings_without_new_episode_known_context': int((warning & known & ~onset).sum()),
            'episode_start_fraction_of_warnings': ratio(hit.sum(), (warning & known).sum()),
            'any_alarm_true_warnings': int((warning & alarm).sum()),
            'original_any_alarm_precision': ratio((warning & alarm).sum(), warning.sum()),
            'original_any_alarm_recall': ratio((warning & alarm).sum(), alarm.sum()),
            'starts_with_records_every_gap_day': int((onset & part.gap_record_coverage.eq('records_every_gap_day')).sum()),
            'starts_with_some_record_free_gap_days': int((onset & ~part.gap_record_coverage.eq('records_every_gap_day')).sum()),
            'starts_with_no_records_in_gap': int((onset & part.gap_record_coverage.eq('no_records_in_gap')).sum())}


def verify_forecast_causality(mart, saved, cfg, old):
    """Rebuild from in-memory adversarially modified future daily records."""
    old_model = lgb.Booster(model_file=str(ROOT / old['model_path']))
    def infer(frame):
        clean = frame.drop(columns=base.TARGET, errors='ignore')
        result = combined.component_scores(clean, cfg)
        result['previous_frozen'] = old_model.predict(clean[old['features']], num_threads=2)
        return result
    scores = infer(mart)
    index = mart.index[mart.date.between(START, END)]
    replay = mart.loc[index, ['object_id', 'date']].copy()
    for model in MODELS:
        replay[model] = scores[model][index]
    joined = saved.merge(replay, on=['object_id', 'date'], validate='one_to_one', suffixes=('_saved', '_replayed'))
    if len(joined) != len(saved) or len(replay) != len(saved):
        raise AssertionError('Replay must retain every original forecast identity')
    replay_differences = {}
    for model in MODELS:
        delta = float(np.max(np.abs(joined[model+'_saved']-joined[model+'_replayed'])))
        if delta > 1e-12:
            raise AssertionError(f'Saved forecast mismatch for {model}: {delta}')
        replay_differences[model] = delta
    cut = pd.Timestamp('2026-03-15')
    real_read = pd.read_parquet
    mutations = {'rows': 0}
    def mutate_future(path, *args, **kwargs):
        frame = real_read(path, *args, **kwargs)
        if Path(path).name.startswith('daily_') and 'date' in frame:
            later = frame.date.gt(cut)
            if later.any():
                frame = frame.copy()
                mutations['rows'] += int(later.sum())
                for column in base.COUNTS:
                    frame.loc[later, column] = frame.loc[later, column]*2+3
                frame.loc[later, 'events_count'] += 100
        return frame
    with patch.object(base.pd, 'read_parquet', side_effect=mutate_future):
        changed, _, _ = base.prepare(end_date=LABEL_END)
    assert changed[['object_id', 'date']].equals(mart[['object_id', 'date']])
    changed_scores = infer(changed)
    earlier = mart.date.le(cut).to_numpy()
    active = mart.date.between(START, cut).to_numpy()
    deltas = {}
    for model in MODELS:
        delta = float(np.max(np.abs(changed_scores[model][earlier]-scores[model][earlier])))
        if delta > 1e-12:
            raise AssertionError(f'Future records changed earlier forecast: {model}, {delta}')
        deltas[model] = delta
    return {'forecast_replay_max_difference': replay_differences,
            'future_records_mutated_after': str(cut.date()), 'mutated_daily_channel_rows': mutations['rows'],
            'causality_checked_history_rows': int(earlier.sum()), 'causality_checked_test_rows': int(active.sum()),
            'future_mutation_max_difference': deltas, 'target_dropped_before_inference': True,
            'mutation_is_in_memory_only': True}


def main():
    started = time.monotonic()
    if not (OUT / 'protocol.md').exists():
        raise RuntimeError('Write the episode-audit protocol before calculation')
    cfg_path = combined.OUT / 'selection.json'; old_path = base.OUT / 'configuration.json'
    cfg = json.loads(cfg_path.read_text()); old = json.loads(old_path.read_text())
    thresholds = {**cfg['thresholds'], 'previous_frozen': old['thresholds']['lightgbm_object']}
    model_paths = [ROOT/old['model_path']]+[ROOT/n['path'] for n in cfg['components'].values() if 'path' in n]
    if checksum(model_paths[0]) != old['model_sha256']:
        raise ValueError('Old model changed')
    for node in cfg['components'].values():
        if 'path' in node and checksum(ROOT/node['path']) != node['sha256']:
            raise ValueError('Selected model artifact changed')
    sources = [PREDICTION, cfg_path, old_path, base.CACHE/'daily_2025.parquet',
               base.CACHE/'daily_2026.parquet', Path(__file__), Path(base.__file__), Path(combined.__file__),
               ROOT/'data/raw/справочник_каналов_датчиков.csv', ROOT/'data/raw/справочник_объектов_диспетчер.csv',
               OUT/'protocol.md', *model_paths]
    initial_hashes = {str(p.relative_to(ROOT)): checksum(p) for p in sources}
    base.dump(OUT/'frozen_inputs.json', {'thresholds': thresholds, 'gaps': GAPS,
              'criterion_changed_only_for_audit': True, 'training_performed': False,
              'threshold_selection_performed': False, 'inputs_sha256': initial_hashes})
    saved = pd.read_parquet(PREDICTION).sort_values(['object_id', 'date']).reset_index(drop=True)
    if len(saved) != 14040 or saved.object_id.nunique()!=78 or saved.date.nunique()!=180:
        raise ValueError('Original audit population changed')
    if saved.date.min()!=START or saved.date.max()!=END or saved[base.TARGET].sum()!=4203:
        raise ValueError('Original dates or target population changed')
    mart, _, coverage = base.prepare(end_date=LABEL_END)
    feature = mart[['object_id','date','observed_today']].rename(columns={'observed_today':'feature_has_records'})
    saved = saved.merge(feature, on=['object_id','date'], how='left', validate='one_to_one')
    if saved.feature_has_records.isna().any():
        raise ValueError('Missing feature-day observation diagnostic')
    all_rows, monthly, observed, labels_checks = [], [], [], []
    labels_by_gap = {}
    for gap in GAPS:
        labels = episode_labels(mart, gap)
        aligned = join_forecasts(saved, labels)
        if not aligned.context_known.all():
            raise AssertionError('Requested historical buffer is incomplete; report rather than silently drop')
        labels_by_gap[gap] = aligned.episode_start.to_numpy()
        # Mutating outcomes after T may alter future labels but never earlier labels.
        label_cut = pd.Timestamp('2026-03-17')
        modified = mart.copy(); later=modified.date.gt(label_cut)
        modified.loc[later, 'alarm_today'] = 1-modified.loc[later, 'alarm_today']
        modified.loc[later, 'observed_today'] = 1
        modified_labels = episode_labels(modified, gap)
        past = labels.date.le(label_cut)
        if not labels.loc[past, 'episode_start'].equals(modified_labels.loc[past, 'episode_start']):
            raise AssertionError('Future mutation changed earlier episode labels')
        labels_checks.append({'gap':gap,'earlier_labels_unchanged_after_mutation':True,
                              'known_context_rows':len(aligned),'future_records_after':str(label_cut.date())})
        for model in MODELS:
            all_rows.append({'gap':gap, **summarize(aligned,model,thresholds[model])})
            for month, part in aligned.groupby(aligned.target_date.dt.to_period('M')):
                monthly.append({'gap':gap,'target_month':str(month),**summarize(part,model,thresholds[model])})
            for field in ['gap_record_coverage','feature_has_records','target_has_records']:
                for value, part in aligned.groupby(field, observed=True, dropna=False):
                    observed.append({'gap':gap,'stratum':field,'stratum_value':str(value),
                                     'stratum_known_at_issue':field=='feature_has_records',
                                     **summarize(part,model,thresholds[model])})
    assert np.all(labels_by_gap[7] <= labels_by_gap[3]) and np.all(labels_by_gap[3] <= labels_by_gap[1])
    for name, rows in [('metrics.csv',all_rows),('monthly.csv',monthly),('observability.csv',observed)]:
        pd.DataFrame(rows).to_csv(OUT/name,index=False)
    inference = verify_forecast_causality(mart, saved, cfg, old)
    changed_files = [name for name,value in initial_hashes.items() if checksum(ROOT/name)!=value]
    if changed_files:
        raise AssertionError('Audit inputs changed: '+str(changed_files))
    check = {'passed':True,'forecast_rows':len(saved),'objects':78,'target_days':180,
             'first_target_date':str((START+pd.Timedelta(days=2)).date()),'last_target_date':str(LABEL_END.date()),
             'fixed_thresholds':thresholds,'label_checks':labels_checks,'gap_nesting_verified':True,
             'warning_partition_verified':True,'input_files_unchanged':True,'source_coverage':coverage,
             **inference,'elapsed_seconds':time.monotonic()-started}
    base.dump(OUT/'checks.json',check)
    write_report(pd.DataFrame(all_rows),pd.DataFrame(monthly),pd.DataFrame(observed),check)
    print(pd.DataFrame(all_rows)[['gap','model','registered_episode_starts','warned_episode_starts','episode_start_recall','repeated_alarm_days','warnings_total','warnings_on_repeated_alarm_days','warnings_on_no_alarm_days']].to_string(index=False),flush=True)


def write_report(metrics, monthly, observed, check):
    def pct(value):return f'{100*value:.2f}%'.replace('.',',')
    rows=['# Начало зарегистрированного эпизода: аудит прежних прогнозов','',
          'Аудит использует те же 14 040 объект-дней и три сохранённых набора прогнозов. '
          'Модели, пороги и состав строк не менялись. Главный gap=1, чувствительности 3 и 7 дней заданы до расчёта.','',
          'Начало на T означает тревожную запись после g полных календарных дней без тревожных записей. '
          'Точный предупреждающий прогноз относится к D=T−2 и выпускается T−1 в 00:00. '
          'Показатель не доказывает новый физический отказ.','',
          '| Модель | Замороженный порог |', '|---|---:|']
    for name in MODELS:
        rows.append(f'| {name} | {check["fixed_thresholds"][name]:.17g} |')
    rows += ['',
          '| Gap | Модель | Начал | Предупреждено | Доля предупреждённых | Повторных тревожных дней |',
          '|---:|---|---:|---:|---:|---:|']
    for r in metrics.itertuples():
        rows.append(f'| {r.gap} | {r.model} | {r.registered_episode_starts} | {r.warned_episode_starts} | {pct(r.episode_start_recall)} | {r.repeated_alarm_days} |')
    rows += ['','## Куда пришлись предупреждения, gap=1','',
             '| Модель | Всего | На начало | На повторную тревогу | Без тревоги | Начало среди предупреждений |',
             '|---|---:|---:|---:|---:|---:|']
    for r in metrics[metrics.gap.eq(1)].itertuples():
        rows.append(f'| {r.model} | {r.warnings_total} | {r.warned_episode_starts} | {r.warnings_on_repeated_alarm_days} | {r.warnings_on_no_alarm_days} | {pct(r.episode_start_fraction_of_warnings)} |')
    main=metrics[(metrics.gap==1)&metrics.model.eq('previous_frozen')].iloc[0]
    rows += ['',f'Прежняя модель демонстрации предупреждает {pct(main.episode_start_recall)} начал против '
             f'{pct(main.original_any_alarm_recall)} всех тревожных объект-дней по старой цели. '
             f'Из её {main.any_alarm_true_warnings} верных предупреждений по прежней цели '
             f'{main.warnings_on_repeated_alarm_days} относятся к повторным тревожным дням.',
             '', 'Предупреждение на повторную тревогу остаётся верным по прежней any-alarm цели. '
             'Оно лишь не совпадает с новой целью начала. Нельзя все предупреждения без нового начала назвать ложными.',
             '', 'Три порога были выбраны для прежней any-alarm цели и дают разное число предупреждений. '
             'Поэтому таблица описывает действующие режимы, а не ранжирует модели по новой цели при одинаковой нагрузке.',
             '', '## Наблюдаемость','',
             'В основной итог включены все календарно определимые начала. Пропущенных календарных контекстов нет. '
             'Наличие хотя бы одной записи не означает полное покрытие всех датчиков. '
             'Страты по предшествующему gap и целевому T являются ретроспективными и не служат правилом отбора при выпуске.', '',
             '| Gap | Начал | Записи в каждый предшествующий день | Часть дней без записей | Во всём gap нет записей |',
             '|---:|---:|---:|---:|---:|']
    for r in metrics[metrics.model.eq('previous_frozen')].itertuples():
        # some_record_free includes the no-record subset; show mutually exclusive counts.
        partial=r.starts_with_some_record_free_gap_days-r.starts_with_no_records_in_gap
        rows.append(f'| {r.gap} | {r.registered_episode_starts} | {r.starts_with_records_every_gap_day} | {partial} | {r.starts_with_no_records_in_gap} |')
    rows += ['', 'Полная разбивка наблюдаемости и показателей находится в `observability.csv`. '
             'Отсутствие записи обозначает только молчание выгрузки, а не исправность.', '',
             '## По целевым месяцам, gap=1','',
             '| Целевой месяц | Модель | Начал | Предупреждено | Доля предупреждённых |',
             '|---|---|---:|---:|---:|']
    for r in monthly[monthly.gap.eq(1)].itertuples():
        rows.append(f'| {r.target_month} | {r.model} | {r.registered_episode_starts} | {r.warned_episode_starts} | {pct(r.episode_start_recall)} |')
    rows += ['', '## Проверка и границы вывода','',
             'Пересчёт замороженного inference воспроизвёл сохранённые прогнозы. '
             'В памяти изменены все суточные записи после 15.03.2026, затем заново построена витрина. '
             'Прогнозы до этой даты не изменились. Целевой столбец перед inference удалён. '
             'Исходные файлы и веса не изменены. Точные отклонения, хеши и числа проверенных строк находятся в `checks.json` и `frozen_inputs.json`.', '',
             'Результат относится к ранее исследованному 2026, не к новому слепому периоду. '
             'Исходная модель обучена на любой тревоге, поэтому аудит новой цели не является переоценкой её заявленного precision. '
             'Сопоставление дней по текущему справочнику и неполная наблюдаемость остаются ограничениями.']
    (OUT/'report.md').write_text('\n'.join(rows)+'\n')


if __name__=='__main__':
    main()
