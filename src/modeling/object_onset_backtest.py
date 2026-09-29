"""One predeclared retrospective transfer check, trained on 2023, evaluated on 2024.

The family was chosen after studying later years. This is a robustness hindcast,
not a fresh blind test or a historically available deployment simulation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

import object_risk_probe as base
import object_onset_experiment as onset
from object_history import prepare_frames
from onset_evaluation import daily_topk, paired_week_bootstrap, summarize

ROOT = base.ROOT
OUT = ROOT / 'reports/object_onset_backtest'
MODELS = ROOT / 'models/object_onset_backtest'
PREDICTIONS = base.CACHE / 'object_onset_backtest_2024.parquet'
START = pd.Timestamp('2023-01-01')
OUTCOMES = [base.TARGET, 'onset_target', 'quiet_intermediate', 'alarm_target',
            'joint_target', 'records_intermediate', 'records_target']


def dictionaries():
    meta = pd.read_csv(ROOT/'data/raw/справочник_каналов_датчиков.csv', dtype=str).rename(
        columns={'ид_канала_данных': 'channel_id', 'ид_объект': 'object_id'})
    objects = pd.read_csv(ROOT/'data/raw/справочник_объектов_диспетчер.csv', dtype=str).rename(
        columns={'ид_объект': 'object_id', 'вид_объекта': 'object_kind', 'родитель': 'parent_id'})
    return meta, objects


def read_daily(end):
    end = pd.Timestamp(end)
    return pd.concat([
        pd.read_parquet(base.CACHE/f'daily_{year}.parquet', columns=['channel_id', 'date', *base.COUNTS],
                        filters=[('date', '>=', START), ('date', '<=', end)])
        for year in range(START.year, end.year+1)
    ], ignore_index=True)


def eligible_objects(raw, meta, objects):
    known = meta.loc[meta.object_id.isin(objects.object_id), ['channel_id', 'object_id']]
    past = raw.loc[raw.date.le('2023-09-28') & raw.events_count.gt(0), ['channel_id']]
    joined = past.merge(known, on='channel_id', how='inner', validate='many_to_one')
    identities = sorted(joined.object_id.unique().tolist())
    if len(identities) < 20:
        raise ValueError('Insufficient fixed population for the predeclared 5/10/20 budgets')
    return identities


def prepare(end, identities, raw=None):
    meta, objects = dictionaries()
    frame, dates, coverage = prepare_frames(
        read_daily(end) if raw is None else raw, meta, objects,
        start_date=START, end_date=end, eligible_object_ids=identities)
    return onset.onset_frame(frame), dates, coverage


def metric(name, frame, values, k=10):
    result = summarize(frame, values, k=k)
    partition = result.pop('warning_partition')
    return {'model': name, **result, **{f'warning_{key}': value for key, value in partition.items()}}


def predict(artifact, frame):
    if artifact['kind'] == 'any_alarm_control':
        return artifact['estimator'].predict_proba(frame[artifact['features']], num_threads=2)[:, 1]
    return onset.predict(artifact, frame)


def save_model(name, artifact):
    path = MODELS/f'{name}.joblib'
    if path.exists():
        raise FileExistsError('Fixed backtest artifact already exists: '+str(path))
    joblib.dump(artifact, path)
    return {'path': str(path.relative_to(ROOT)), 'sha256': base.checksum(path),
            'iterations': artifact.get('iterations'), 'features': artifact['features']}


def source_hashes():
    paths = [Path(__file__), OUT/'protocol.md', ROOT/'src/modeling/object_history.py',
             ROOT/'src/modeling/object_risk_probe.py', ROOT/'src/modeling/object_onset_experiment.py',
             ROOT/'src/modeling/object_onset_statistics.py', ROOT/'src/modeling/onset_evaluation.py',
             onset.OLD_CFG, base.CACHE/'daily_2023.parquet', base.CACHE/'daily_2023.history_manifest.json',
             ROOT/'data/raw/справочник_каналов_датчиков.csv',
             ROOT/'data/raw/справочник_объектов_диспетчер.csv']
    return {str(path.relative_to(ROOT)): base.checksum(path) for path in paths}


def load_models(cfg):
    for path, expected in cfg['sources'].items():
        if base.checksum(ROOT/path) != expected:
            raise ValueError('Frozen experiment source changed: '+path)
    artifacts = {}
    for name, node in cfg['models'].items():
        if base.checksum(ROOT/node['path']) != node['sha256']:
            raise ValueError('Frozen model changed: '+name)
        artifacts[name] = joblib.load(ROOT/node['path'])
    return artifacts


def fit():
    if (OUT/'configuration.json').exists() or MODELS.exists():
        raise FileExistsError('This fixed experiment was already fitted; no silent retuning')
    started = time.monotonic()
    meta, objects = dictionaries()
    raw = read_daily('2023-11-30')
    identities = eligible_objects(raw, meta, objects)
    frame, dates, coverage = prepare('2023-11-30', identities, raw)
    train = frame.loc[frame.date.le('2023-09-28')]
    tune = frame.loc[frame.date.between('2023-10-01', '2023-11-28')]
    refit = frame.loc[frame.date.le('2023-11-28')]
    for split, last_label in [(train, '2023-09-30'), (tune, '2023-11-30'), (refit, '2023-11-30')]:
        assert split.onset_target.notna().all()
        assert dates.loc[split.index].max() <= pd.Timestamp(last_label)
    selected = onset.fit_candidate('hurdle_15', train, tune)
    columns = json.loads(onset.OLD_CFG.read_text())['features']
    parameters = dict(objective='binary', metric='average_precision', n_estimators=400,
                      num_leaves=15, learning_rate=.04, min_child_samples=30, reg_lambda=10,
                      colsample_bytree=.85, random_state=84, n_jobs=2, verbosity=-1)
    control = lgb.LGBMClassifier(**parameters)
    control.fit(train[columns], train[base.TARGET].astype(int), categorical_feature=onset.CATS,
                eval_set=[(tune[columns], tune[base.TARGET].astype(int))],
                callbacks=[lgb.early_stopping(40, first_metric_only=True, verbose=False), lgb.log_evaluation(0)])
    control_iterations = int(control.best_iteration_ or control.n_estimators)
    initial = {'hurdle_15': selected,
               'any_alarm_control': {'kind': 'any_alarm_control', 'estimator': control, 'features': columns},
               'frequency': onset.fit_candidate('frequency', train)}
    pd.DataFrame([metric(name, tune, predict(model, tune)) for name, model in initial.items()]).to_csv(
        OUT/'tuning_metrics.csv', index=False)
    # Refit only after the two early-stopping budgets are fixed; no December data read.
    final_hurdle = onset.fit_candidate('hurdle_15', refit, iterations=selected['iterations'])
    final_control = lgb.LGBMClassifier(**{**parameters, 'n_estimators': control_iterations})
    final_control.fit(refit[columns], refit[base.TARGET].astype(int), categorical_feature=onset.CATS)
    MODELS.mkdir(parents=True)
    models = {
        'hurdle_15': save_model('hurdle_15', final_hurdle),
        'any_alarm_control': save_model('any_alarm_control', {'kind': 'any_alarm_control',
            'estimator': final_control, 'features': columns, 'iterations': [control_iterations]}),
        'frequency': save_model('frequency', onset.fit_candidate('frequency', refit)),
    }
    cfg = {'created_at': pd.Timestamp.now(tz='UTC').isoformat(), 'fixed_family': 'hurdle_15',
           'primary_k': 10, 'eligible_object_ids': identities, 'population_cutoff': '2023-09-28',
           'models': models, 'sources': source_hashes(), 'coverage_through_november': coverage,
           'train_rows': len(train), 'tune_rows': len(tune), 'refit_rows': len(refit),
           'train_label_end': '2023-09-30', 'refit_label_end': '2023-11-30',
           'december_read_during_fit': False, 'evaluation_2024_read_during_fit': False,
           'metadata_point_in_time': False, 'model_family_chosen_after_2026': True,
           'elapsed_seconds': time.monotonic()-started}
    base.dump(OUT/'configuration.json', cfg)
    print(json.dumps({'objects': len(identities), 'iterations': {n: x['iterations'] for n, x in models.items()},
                      'elapsed_seconds': cfg['elapsed_seconds']}, ensure_ascii=False), flush=True)


def evaluate():
    if (OUT/'checks.json').exists():
        raise FileExistsError('Fixed evaluation exists; preserve the original result')
    cfg = json.loads((OUT/'configuration.json').read_text())
    models = load_models(cfg)
    identities = cfg['eligible_object_ids']
    history, _, _ = prepare('2023-12-31', identities)
    december = history.loc[history.date.between('2023-12-01', '2023-12-29')]
    dec_rows = [metric(name, december, predict(model, december)) for name, model in models.items()]
    pd.DataFrame(dec_rows).to_csv(OUT/'december_metrics.csv', index=False)
    by_name = {x['model']: x for x in dec_rows}
    gate = by_name['hurdle_15']['precision_at_k'] >= by_name['any_alarm_control']['precision_at_k']
    base.dump(OUT/'december_gate.json', {'passed': gate, 'tuning_performed': False,
                                      '2024_read_at_gate': False})
    frame, _, coverage = prepare('2024-06-30', identities)
    test = frame.loc[frame.date.between('2023-12-31', '2024-06-28')].copy()
    assert test.date.nunique() == 181 and len(test) == 181*len(identities)
    scores = {name: predict(model, test) for name, model in models.items()}
    rows = [metric(name, test, score, k) for k in [5, 10, 20] for name, score in scores.items()]
    pd.DataFrame(rows).to_csv(OUT/'evaluation_metrics.csv', index=False)
    monthly = []
    for month, part in test.groupby((test.date+pd.Timedelta(days=2)).dt.strftime('%Y-%m')):
        positions = test.index.get_indexer(part.index)
        for name, score in scores.items():
            monthly.append({'target_month': month, **metric(name, part, score[positions])})
    pd.DataFrame(monthly).to_csv(OUT/'monthly_metrics.csv', index=False)
    strata = []
    for name, score in scores.items():
        warnings = daily_topk(test, score, k=10)
        for observed in [0, 1]:
            mask = test.records_intermediate.eq(observed).to_numpy()
            target = test.onset_target.eq(1).to_numpy()
            strata.append({'model': name, 'records_intermediate': observed, 'rows': int(mask.sum()),
                'onsets': int((target & mask).sum()), 'warnings': int((warnings & mask).sum()),
                'tp': int((warnings & mask & target).sum()), 'retrospective_filter_only': True})
    pd.DataFrame(strata).to_csv(OUT/'observability.csv', index=False)
    uncertainty = paired_week_bootstrap(test, scores['hurdle_15'], scores['any_alarm_control'], k=10, n=1000, seed=84)
    base.dump(OUT/'uncertainty.json', uncertainty)
    no_outcomes = {name: float(np.max(np.abs(predict(model, test.drop(columns=OUTCOMES))-scores[name])))
                   for name, model in models.items()}
    prefix, _, _ = prepare('2024-03-15', identities)
    truncated = prefix.loc[prefix.date.between('2023-12-31', '2024-03-15')].copy()
    original = test.loc[test.date.le('2024-03-15')]
    pd.testing.assert_frame_equal(original[['object_id', 'date', *[c for c in onset.features() if c != 'object_id']]].reset_index(drop=True),
                                  truncated[['object_id', 'date', *[c for c in onset.features() if c != 'object_id']]].reset_index(drop=True), check_exact=True)
    positions = test.index.get_indexer(original.index)
    prefix_deltas = {name: float(np.max(np.abs(predict(model, truncated.drop(columns=OUTCOMES))-scores[name][positions])))
                     for name, model in models.items()}
    assert max([*no_outcomes.values(), *prefix_deltas.values()]) == 0
    output = test[['object_id', 'date', 'onset_target', 'alarm_target', 'records_intermediate']].copy()
    for name, score in scores.items(): output[name] = score
    output.to_parquet(PREDICTIONS, index=False)
    delta = uncertainty['delta_precision_at_k']
    passed = gate and delta['estimate'] > 0 and delta['ci95'][0] > 0
    base.dump(OUT/'checks.json', {'december_gate_passed': gate, 'robustness_criterion_passed': passed,
        'rows': len(test), 'objects': len(identities), 'days': 181, 'onsets': int(test.onset_target.sum()),
        'no_outcomes_max_score_difference': no_outcomes, 'truncated_history_max_score_difference': prefix_deltas,
        'feature_prefix_exactly_equal': True, 'evaluation_coverage': coverage,
        'configuration_sha256': base.checksum(OUT/'configuration.json'),
        'daily_2024_sha256': base.checksum(base.CACHE/'daily_2024.parquet'),
        'predictions_sha256': base.checksum(PREDICTIONS), 'blind_test': False,
        'production_model_replaced': False, 'metadata_point_in_time': False})
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(json.dumps({'december_gate': gate, 'delta_precision_at_10': delta,
                      'robustness_criterion_passed': passed}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['fit', 'evaluate'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT/'protocol.md').exists(): raise FileNotFoundError('Protocol must precede experiment')
    globals()[args.phase]()
