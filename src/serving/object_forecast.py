"""Generate immutable historical forecast runs from available daily snapshots.

No future-outcome archive is read. Batch timestamps describe a replay clock,
not a claimed connection to the operational monitoring system.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/modeling'))
sys.path.insert(0, str(ROOT))
import object_risk_probe as features
from export_object_demo import LABELS, forecast_times, value_text

# The frozen threshold used target outcomes through 2025-11-30 inclusive.
# This is a retrospective model-availability boundary, not its real creation time.
MODEL_AVAILABLE_AT = '2025-12-01T00:00:00+03:00'


@dataclass(frozen=True)
class DailyBatch:
    day: str
    available_at: str
    rows: pd.DataFrame


def aware(value):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError('Availability and issue timestamps must include a timezone')
    return stamp.tz_convert('Europe/Moscow')


def require_model_available(feature_date):
    if aware(forecast_times(feature_date)['issue_time']) < aware(MODEL_AVAILABLE_AT):
        raise ValueError('Frozen model/threshold unavailable before completion of November 2025 selection')


def validate_daily(frame):
    columns = ['channel_id', 'date', *features.COUNTS]
    if set(columns) - set(frame):
        raise ValueError('Incomplete daily feature schema')
    result = frame[columns].copy()
    if result.channel_id.isna().any() or result.channel_id.astype(str).str.strip().eq('').any():
        raise ValueError('Missing channel identifier')
    result['channel_id'] = result.channel_id.astype(str)
    result['date'] = pd.to_datetime(result.date, errors='raise')
    if result.date.dt.tz is not None or result.date.isna().any() or not result.date.eq(result.date.dt.normalize()).all():
        raise ValueError('Expected valid, timezone-free calendar dates')
    if result.duplicated(['channel_id', 'date']).any():
        raise ValueError('Duplicate channel-day rows in snapshot')
    for col in features.COUNTS:
        values = pd.to_numeric(result[col], errors='raise').to_numpy(dtype='float64')
        if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError('Daily counts must be finite nonnegative integers')
        result[col] = values
    if (result[features.COUNTS[1:]].gt(result.events_count, axis=0)).any().any():
        raise ValueError('A component count exceeds the number of events')
    return result.sort_values(['channel_id', 'date']).reset_index(drop=True)


def fingerprint(frame):
    """Stable content hash of the selected counts, independent of row order."""
    ordered = frame.sort_values(['channel_id', 'date'])
    h = hashlib.sha256(b'collector-daily-schema-v1\0')
    for channel in ordered.channel_id.astype(str):
        encoded = channel.encode('utf-8')
        h.update(len(encoded).to_bytes(4, 'little')); h.update(encoded)
    h.update(ordered.date.to_numpy(dtype='datetime64[ns]').astype('<i8').tobytes())
    h.update(ordered[features.COUNTS].to_numpy(dtype='<f8').tobytes())
    return h.hexdigest()


def assemble_history(seed, seed_through, batches, feature_date):
    """Pick the latest complete daily version that was available at issue time."""
    cutoff = pd.Timestamp(feature_date).normalize()
    seed_end = pd.Timestamp(seed_through).normalize()
    issue = aware(forecast_times(str(cutoff.date()))['issue_time'])
    seed = validate_daily(seed)
    if seed_end >= cutoff or seed.date.gt(seed_end).any() or seed.date.lt('2025-01-01').any():
        raise ValueError('Bootstrap history must end before the first replay feature day')
    visible = {}; accepted = []; versions = {}
    for batch in batches:
        day = pd.Timestamp(batch.day).normalize()
        available = aware(batch.available_at)
        if day <= seed_end:
            raise ValueError('A daily update cannot overlap the immutable bootstrap')
        if available < day.tz_localize('Europe/Moscow') + pd.Timedelta(days=1):
            raise ValueError('A complete day cannot be available before its end')
        if day > cutoff or available > issue:
            continue
        rows = validate_daily(batch.rows)
        if not rows.date.eq(day).all():
            raise ValueError('A daily snapshot contains another date')
        digest = fingerprint(rows)
        version_key = (day, available)
        if version_key in versions and versions[version_key] != digest:
            raise ValueError('Conflicting snapshots at the same availability timestamp')
        versions[version_key] = digest
        prior = visible.get(day)
        if prior and available == prior[0] and digest != prior[1]:
            raise ValueError('Conflicting snapshots at the same availability timestamp')
        if prior is None or available > prior[0]:
            visible[day] = (available, digest, rows)
    if cutoff not in visible:
        raise ValueError('No complete, available snapshot for the feature day')
    for day, (available, digest, _) in sorted(visible.items()):
        accepted.append({'day': str(day.date()), 'available_at': available.isoformat(), 'sha256': digest})
    joined = pd.concat([seed, *[v[2] for _, v in sorted(visible.items())]], ignore_index=True)
    missing = pd.date_range(seed_end+pd.Timedelta(days=1), cutoff).difference(pd.DatetimeIndex(visible))
    # Missing a full input package is an ingestion gap, not an empty event day.
    if len(missing):
        raise ValueError('Missing complete daily snapshots: '+', '.join(str(d.date()) for d in missing))
    joined = joined.sort_values(['channel_id', 'date']).reset_index(drop=True)
    return joined, {'bootstrap_through': str(seed_end.date()), 'bootstrap_sha256': fingerprint(seed),
                    'accepted_batches': accepted, 'input_sha256': fingerprint(joined)}


def load_frozen_inputs():
    cfg = json.loads((ROOT / 'reports/object_risk_probe/configuration.json').read_text())
    model_path = ROOT / cfg['model_path']
    if features.checksum(model_path) != cfg['model_sha256']:
        raise ValueError('Frozen model checksum changed')
    meta_path = ROOT / 'data/raw/справочник_каналов_датчиков.csv'
    object_path = ROOT / 'data/raw/справочник_объектов_диспетчер.csv'
    if features.checksum(meta_path) != cfg['meta_sha256'] or features.checksum(object_path) != cfg['objects_sha256']:
        raise ValueError('Dictionary differs from the frozen model input')
    meta = pd.read_csv(meta_path, dtype=str).rename(columns={'ид_канала_данных': 'channel_id', 'ид_объект': 'object_id'})
    objects = pd.read_csv(object_path, dtype=str).rename(columns={'ид_объект': 'object_id', 'вид_объекта': 'object_kind', 'родитель': 'parent_id'})
    model = lgb.Booster(model_file=str(model_path))
    if model.feature_name() != cfg['features']:
        raise ValueError('Frozen feature schema changed')
    return cfg, model, meta, objects


def predict_payload(history, provenance, feature_date, cfg, model, meta, objects):
    require_model_available(feature_date)
    times = forecast_times(feature_date)
    mart, _, coverage = features.prepare_frames(history, meta, objects, pd.Timestamp(feature_date))
    latest = mart.loc[mart.date.eq(pd.Timestamp(feature_date))].copy()
    if not latest[features.TARGET].isna().all() or features.TARGET in cfg['features']:
        raise ValueError('Unknown future labels must be absent from inference')
    x = latest[cfg['features']]
    scores = model.predict(x, num_threads=2)
    terms = model.predict(x, pred_contrib=True, num_threads=2)
    threshold = float(cfg['thresholds']['lightgbm_object'])
    implementation = hashlib.sha256(Path(__file__).read_bytes()+Path(features.__file__).read_bytes()).hexdigest()
    lineage = {'input_sha256': provenance['input_sha256'], 'model_sha256': cfg['model_sha256'],
               'configuration_sha256': features.checksum(ROOT / 'reports/object_risk_probe/configuration.json'),
               'implementation_sha256': implementation, 'batches': provenance['accepted_batches'],
               'feature_date': feature_date}
    run_hash = hashlib.sha256(json.dumps(lineage, sort_keys=True).encode()).hexdigest()
    run_id = f'object_{feature_date}_{run_hash[:16]}'
    catalog = objects.set_index('object_id')
    cards = []
    recent = mart.loc[mart.date.between(pd.Timestamp(feature_date)-pd.Timedelta(days=6), pd.Timestamp(feature_date))]
    alarm_days = recent.groupby('object_id', observed=True).alarm_today.sum()
    observed_days = recent.groupby('object_id', observed=True).observed_today.sum()
    for i, (_, row) in enumerate(latest.iterrows()):
        oid = str(row.object_id); obj = catalog.loc[oid]; pid = str(obj.parent_id)
        positive = [int(j) for j in np.argsort(terms[i, :-1])[::-1] if terms[i, j] > 0][:3]
        observations = int(row.observed_channels)
        cards.append({'id': f'{run_id}_{oid}', 'entity_mode': 'object', 'object_id': oid,
            'object_name': str(obj['диспетчерское_название_объекта']), 'parent_id': pid,
            'parent_name': str(catalog.loc[pid, 'диспетчерское_название_объекта']) if pid in catalog.index else f'Группа {pid}',
            'object_kind': str(row.object_kind), 'risk_type': 'Зарегистрированная тревога на объекте',
            'score': float(scores[i]), 'warning': bool(scores[i] >= threshold), 'current_alarm': bool(row.alarm_today),
            'events_today': int(row.events_count), 'alarm_channels': int(row.alarm_channels),
            'observed_channels': observations, 'catalog_channels': int(row.catalog_channels),
            'alarm_days_7d': int(alarm_days.loc[row.object_id]), 'observed_days_7d': int(observed_days.loc[row.object_id]),
            'observation_note': 'Нет записей за день; исправность не подтверждена.' if not observations else 'Наличие записей не гарантирует полное покрытие наблюдения.',
            'explanation': [{'feature': cfg['features'][j], 'label': LABELS.get(cfg['features'][j], cfg['features'][j]),
                             'value': value_text(x.iloc[i, j]), 'contribution_log_odds': float(terms[i, j])} for j in positive],
            'suggested_action': 'Сопоставить доступность данных, журналы и плановые работы; диспетчер определяет необходимость осмотра.'})
    cards.sort(key=lambda card: (-card['score'], card['object_id']))
    payload = {'run_id': run_id, 'entity_mode': 'object', 'mode': 'historical_replay', **times,
        'model': 'LightGBM · сохранённая объектная модель', 'model_sha256': cfg['model_sha256'],
        'input_sha256': provenance['input_sha256'], 'implementation_sha256': implementation,
        'configuration_sha256': lineage['configuration_sha256'], 'input_provenance': provenance,
        'threshold': threshold, 'score_kind': 'uncalibrated_model_score', 'total_objects': len(cards),
        'shown_cards': len(cards), 'warnings_count': sum(c['warning'] for c in cards),
        'observed_objects_today': sum(c['observed_channels'] > 0 for c in cards),
        'limitations': 'Историческое воспроизведение дневных пакетов. Тревога не равна поломке. Score не калиброван. Рабочий поток СМВУ не подключён.',
        'scheme_note': 'Схема групп справочника, без географических координат.', 'cards': cards}
    return payload, latest, coverage


def read_daily(start, end):
    result = pd.concat([pd.read_parquet(ROOT / f'data/interim/daily_{year}.parquet',
        columns=['channel_id', 'date', *features.COUNTS], filters=[('date', '>=', pd.Timestamp(start)), ('date', '<=', pd.Timestamp(end))])
        for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year+1)], ignore_index=True)
    if result.empty:
        raise ValueError('No confirmed archive package for the requested dates; an empty query is not a complete empty day')
    return result


def replay(first, last, database, output):
    from api.forecast_store import publish_run
    require_model_available(first)
    cfg, model, meta, objects = load_frozen_inputs()
    first_day, last_day = pd.Timestamp(first), pd.Timestamp(last)
    if first_day > last_day or first_day <= pd.Timestamp('2025-01-01'):
        raise ValueError('Invalid replay range')
    seed_end = first_day-pd.Timedelta(days=1)
    seed = read_daily('2025-01-01', str(seed_end.date()))
    batches=[]; results=[]
    for day in pd.date_range(first_day, last_day):
        start=time.perf_counter()
        date=str(day.date()); batch=DailyBatch(date, forecast_times(date)['issue_time'], read_daily(date,date))
        batches.append(batch)
        loaded=time.perf_counter()
        history, lineage=assemble_history(seed,str(seed_end.date()),batches,date)
        assembled=time.perf_counter()
        payload, latest, coverage=predict_payload(history,lineage,date,cfg,model,meta,objects)
        predicted=time.perf_counter()
        saved=publish_run(database,payload)
        end=time.perf_counter()
        repeated=publish_run(database,payload)
        assert saved['run_id']==repeated['run_id']
        results.append({'feature_date':date,'run_id':payload['run_id'],'objects':len(latest),
            'read_day_seconds':loaded-start,'assemble_seconds':assembled-loaded,
            'features_and_predict_seconds':predicted-assembled,'publish_seconds':end-predicted,
            'batch_to_archive_seconds':end-start,'duplicate_publish_idempotent':True,'coverage':coverage})
        print(json.dumps(results[-1]),flush=True)
    output.mkdir(parents=True,exist_ok=True)
    features.dump(output/'runs.json', {'mode':'historical_replay','bootstrap_read_excluded_from_day_latency':True,
        'model_load_excluded_from_day_latency':True,'no_future_outcome_archive_read':True,'runs':results})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first-feature-date',default='2026-06-26'); p.add_argument('--last-feature-date',default='2026-06-28')
    p.add_argument('--database',type=Path,default=ROOT/'data/app/tickets.sqlite3')
    p.add_argument('--report-dir',type=Path,default=ROOT/'reports/forecast_replay')
    args=p.parse_args(); replay(args.first_feature_date,args.last_feature_date,args.database,args.report_dir)
