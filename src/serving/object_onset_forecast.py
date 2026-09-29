"""Publish a separate historical candidate queue for registered episode starts.

The November-selected model and daily top-10 policy are frozen. December is a
confirmation gate, so this replay starts no earlier than issue 2026-01-01.
No 2026 outcome, metric, or promotion decision is read by this adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / 'src/modeling'))
sys.path.insert(0, str(ROOT))
import object_forecast as daily
import object_onset_experiment as experiment
from export_object_demo import forecast_times, value_text

REFIT_PATH = ROOT / 'reports/object_onset_experiment/refit.json'
REFIT_SHA256 = '1cc105c2c2cca9cd216e42a5e4385d8b06515e1992bdf0ada0e97ff2ede3e3ee'
TARGET_KIND = 'registered_episode_start_g1'
TARGET_DEFINITION = ('Начало зарегистрированной серии: в день D+1 нет тревожной записи, '
                     'в день D+2 есть тревожная запись. Данные известны только по D включительно; '
                     'отсутствие записей не подтверждает исправность.')
POLICY = {'kind': 'daily_top_k', 'k': 10, 'tie_break': 'score_desc_object_id_asc'}
MODEL_FIT_AVAILABLE_AT = '2025-12-01T00:00:00+03:00'
DECEMBER_GATE_AVAILABLE_AT = '2026-01-01T00:00:00+03:00'
FUTURE_COLUMNS = [daily.features.TARGET, 'onset_target', 'alarm_target', 'joint_target',
                  'quiet_intermediate', 'records_intermediate', 'records_target']


def require_candidate_available(feature_date):
    if daily.aware(forecast_times(feature_date)['issue_time']) < daily.aware(DECEMBER_GATE_AVAILABLE_AT):
        raise ValueError('Post-December-gate candidate unavailable before 2026-01-01 issue')


def load_frozen_inputs():
    # Pin the manifest before loading its local joblib artifact.
    if daily.features.checksum(REFIT_PATH) != REFIT_SHA256:
        raise ValueError('Frozen onset refit configuration changed')
    cfg = json.loads(REFIT_PATH.read_text())
    experiment.verify_sources(cfg)
    if daily.features.checksum(experiment.OUT / 'selection.json') != cfg['selection_sha256']:
        raise ValueError('Frozen onset selection changed')
    selection = json.loads((experiment.OUT / 'selection.json').read_text())
    if (cfg['parts'] != {'hurdle_15': 1.0} or selection['parts'] != cfg['parts']
            or selection['primary_k'] != POLICY['k'] or not cfg['december_gate_passed']):
        raise ValueError('Candidate or warning policy differs from the frozen selection')
    node = cfg['models']['hurdle_15']
    artifact = experiment.load_artifact(node)
    if (artifact['kind'] != 'hurdle' or len(artifact['models']) != 2
            or artifact['features'] != selection['features']
            or set(artifact['features']).intersection(FUTURE_COLUMNS)):
        raise ValueError('Frozen onset feature or model schema changed')
    for model in artifact['models']:
        if model.booster_.feature_name() != artifact['features']:
            raise ValueError('Component feature order differs from the frozen schema')
    meta = pd.read_csv(ROOT / 'data/raw/справочник_каналов_датчиков.csv', dtype=str).rename(
        columns={'ид_канала_данных': 'channel_id', 'ид_объект': 'object_id'})
    objects = pd.read_csv(ROOT / 'data/raw/справочник_объектов_диспетчер.csv', dtype=str).rename(
        columns={'ид_объект': 'object_id', 'вид_объекта': 'object_kind', 'родитель': 'parent_id'})
    return cfg, artifact, meta, objects


def rank_cards(cards):
    """Exactly ten, including tied scores; no future label or current-alarm filter."""
    if len(cards) < POLICY['k']:
        raise ValueError('Fewer objects than the frozen daily warning budget')
    ids = [str(card['object_id']) for card in cards]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate object id in ranking')
    if any(not np.isfinite(card['score']) or not 0 <= card['score'] <= 1 for card in cards):
        raise ValueError('Invalid onset score')
    cards.sort(key=lambda card: (-card['score'], str(card['object_id'])))
    for rank, card in enumerate(cards, start=1):
        card.update(rank=rank, warning=rank <= POLICY['k'])
    return float(cards[POLICY['k'] - 1]['score'])


def predict_payload(history, provenance, feature_date, cfg, artifact, meta, objects):
    require_candidate_available(feature_date)
    times = forecast_times(feature_date)
    cutoff = pd.Timestamp(feature_date)
    history = daily.validate_daily(history)
    if history.date.gt(cutoff).any() or history.date.lt('2025-01-01').any():
        raise ValueError('History contains future or pre-bootstrap rows')
    if daily.fingerprint(history) != provenance['input_sha256']:
        raise ValueError('Input history differs from its provenance')
    visible = provenance['accepted_batches']
    if not any(batch['day'] == feature_date for batch in visible):
        raise ValueError('Feature day lacks a confirmed complete input package')
    for batch in visible:
        at = daily.aware(batch['available_at'])
        if (pd.Timestamp(batch['day']) > cutoff or at > daily.aware(times['issue_time'])
                or at < pd.Timestamp(batch['day']).tz_localize('Europe/Moscow') + pd.Timedelta(days=1)):
            raise ValueError('Batch unavailable at the historical issue time')
    mart, _, coverage = daily.features.prepare_frames(history, meta, objects, cutoff)
    mart = experiment.onset_frame(mart)
    latest = mart.loc[mart.date.eq(cutoff)].copy()
    if latest.empty or latest.object_id.duplicated().any() or not latest[FUTURE_COLUMNS].isna().all().all():
        raise ValueError('Expected unique current objects without future labels')
    # Remove every derived future column before calling the frozen predictor.
    latest = latest.drop(columns=FUTURE_COLUMNS)
    scores = experiment.predict(artifact, latest)
    implementation = hashlib.sha256(b''.join(Path(path).read_bytes() for path in
        [__file__, daily.__file__, daily.features.__file__, experiment.__file__,
         ROOT / 'src/modeling/export_object_demo.py'])).hexdigest()
    model_sha = cfg['models']['hurdle_15']['sha256']
    lineage = {'target_kind': TARGET_KIND, 'warning_policy': POLICY,
               'input_sha256': provenance['input_sha256'], 'input_provenance': provenance,
               'model_sha256': model_sha, 'configuration_sha256': REFIT_SHA256,
               'implementation_sha256': implementation, 'feature_date': feature_date}
    digest = hashlib.sha256(json.dumps(lineage, sort_keys=True).encode()).hexdigest()
    run_id = f'onset_{feature_date}_{digest[:16]}'
    catalog = objects.set_index('object_id')
    recent = mart.loc[mart.date.between(cutoff-pd.Timedelta(days=6), cutoff)]
    alarm_days = recent.groupby('object_id', observed=True).alarm_today.sum()
    observed_days = recent.groupby('object_id', observed=True).observed_today.sum()
    cards = []
    for score, (_, row) in zip(scores, latest.iterrows(), strict=True):
        oid = str(row.object_id); obj = catalog.loc[oid]; pid = str(obj.parent_id)
        observations = int(row.observed_channels)
        cards.append({'id': f'{run_id}_{oid}', 'entity_mode': 'object', 'object_id': oid,
            'object_name': str(obj['диспетчерское_название_объекта']), 'parent_id': pid,
            'parent_name': str(catalog.loc[pid, 'диспетчерское_название_объекта']) if pid in catalog.index else f'Группа {pid}',
            'object_kind': str(row.object_kind), 'risk_type': 'Начало зарегистрированной серии тревог',
            'score': float(score), 'current_alarm': bool(row.alarm_today),
            'events_today': int(row.events_count), 'alarm_channels': int(row.alarm_channels),
            'observed_channels': observations, 'catalog_channels': int(row.catalog_channels),
            'alarm_days_7d': int(alarm_days.loc[row.object_id]), 'observed_days_7d': int(observed_days.loc[row.object_id]),
            'observation_note': 'Нет записей за день; исправность не подтверждена.' if not observations else 'Наличие записей не гарантирует полное покрытие наблюдения.',
            'explanation_kind': 'observations',
            'explanation': [{'feature': field, 'label': label, 'value': value_text(row[field])} for field, label in [
                ('onset_mean28', 'Доля дней с началом зарегистрированной серии за 28 дней'),
                ('days_since_onset', 'Дней с последнего начала серии (−1: не было в истории)'),
                ('observed_channels', 'Каналы с записями в день признаков')]],
            'suggested_action': 'Проверить доступность наблюдений и контекст журнала; диспетчер решает, нужен ли осмотр.'})
    boundary = rank_cards(cards)
    payload = {'run_id': run_id, 'entity_mode': 'object', 'mode': 'historical_replay', **times,
        'target_kind': TARGET_KIND, 'target_definition': TARGET_DEFINITION,
        'warning_policy': dict(POLICY), 'candidate_status': 'research_candidate',
        'model_fit_available_at': MODEL_FIT_AVAILABLE_AT, 'december_gate_available_at': DECEMBER_GATE_AVAILABLE_AT,
        'model': 'Двухэтапная модель начала серии · исследовательский кандидат',
        'model_sha256': model_sha, 'input_sha256': provenance['input_sha256'],
        'configuration_sha256': REFIT_SHA256, 'implementation_sha256': implementation,
        'input_provenance': provenance, 'threshold': boundary, 'threshold_role': 'rank_boundary_score_only',
        'score_kind': 'uncalibrated_model_score', 'total_objects': len(cards), 'shown_cards': len(cards),
        'warnings_count': sum(card['warning'] for card in cards),
        'observed_objects_today': sum(card['observed_channels'] > 0 for card in cards),
        'limitations': ('Историческое воспроизведение кандидата, не действующий поток. '
            'День D+1 неизвестен при выпуске. Ровно 10 объектов ежедневно; повторные предупреждения не подавляются. '
            'Score не калиброван и не является вероятностью физической поломки. '
            'Дата записи выпуска в архив отделена от условного исторического времени выпуска.'),
        'scheme_note': 'Схема групп справочника, без географических координат.', 'cards': cards}
    return payload, latest, coverage


def replay(first, last, database, output):
    from api.forecast_store import publish_run
    require_candidate_available(first)
    first_day, last_day = pd.Timestamp(first), pd.Timestamp(last)
    if first_day > last_day:
        raise ValueError('Invalid replay range')
    cfg, artifact, meta, objects = load_frozen_inputs()
    seed_end = first_day-pd.Timedelta(days=1)
    seed = daily.read_daily('2025-01-01', str(seed_end.date()))
    batches = []; results = []
    for day in pd.date_range(first_day, last_day):
        started = time.perf_counter(); date = str(day.date())
        batches.append(daily.DailyBatch(date, forecast_times(date)['issue_time'], daily.read_daily(date, date)))
        history, provenance = daily.assemble_history(seed, str(seed_end.date()), batches, date)
        payload, latest, coverage = predict_payload(history, provenance, date, cfg, artifact, meta, objects)
        predicted = time.perf_counter()
        saved = publish_run(database, payload)
        finished = time.perf_counter()
        if publish_run(database, payload) != saved:
            raise AssertionError('Publication is not idempotent')
        result = {'feature_date': date, 'run_id': payload['run_id'], 'objects': len(latest),
                  'warnings_count': payload['warnings_count'], 'daily_input_to_prediction_seconds': predicted-started,
                  'publish_seconds': finished-predicted, 'daily_input_to_archive_seconds': finished-started,
                  'duplicate_publish_idempotent': True, 'coverage': coverage}
        results.append(result); print(json.dumps(result), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    daily.features.dump(output/'runs.json', {'mode': 'historical_replay', 'candidate_status': 'research_candidate',
        'target_kind': TARGET_KIND, 'warning_policy': POLICY, 'configuration_sha256': REFIT_SHA256,
        'model_load_and_bootstrap_read_excluded_from_daily_latency': True,
        'no_2026_outcome_metric_or_promotion_decision_read': True, 'runs': results})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first-feature-date', default='2026-06-26')
    parser.add_argument('--last-feature-date', default='2026-06-28')
    parser.add_argument('--database', type=Path, default=ROOT/'data/app/onset_demo.sqlite3')
    parser.add_argument('--report-dir', type=Path, default=ROOT/'reports/onset_demo')
    args = parser.parse_args()
    replay(args.first_feature_date, args.last_feature_date, args.database, args.report_dir)
