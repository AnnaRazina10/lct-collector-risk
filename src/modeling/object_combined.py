"""Select a bounded mixture of independent object forecasting families."""
import argparse
import json
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import object_risk_probe as base
import object_statistical_candidate as statistical_model

OUT = base.ROOT / 'reports/object_combined'
STAT = base.ROOT / 'reports/object_statistical_candidate'
SEQUENCE = base.ROOT / 'reports/object_sequence_candidate'


def select():
    statistical = pd.read_parquet(base.CACHE / 'object_statistical_tuning.parquet')
    sequence = pd.read_parquet(SEQUENCE / 'predictions/tune.parquet')
    keys = ['object_id', 'date', base.TARGET]
    gru_columns = [c for c in sequence if c.startswith('gru_')]
    pred = statistical.merge(sequence[keys+gru_columns], on=keys, validate='one_to_one')
    if len(pred) != len(statistical) or len(pred) != len(sequence):
        raise ValueError('Tuning identities or labels differ')
    y = pred[base.TARGET].to_numpy()
    selected = ['lightgbm_control']
    for family in ['markov_', 'seasonal_', 'linear_', 'spline_', 'gru_']:
        selected.append(max([c for c in pred if c.startswith(family)], key=lambda c: average_precision_score(y, pred[c])))
    stat = max(selected[1:-1], key=lambda c: average_precision_score(y, pred[c]))
    gru = selected[-1]
    graphs = {name: {name: 1.0} for name in selected}
    for name in selected[1:]:
        for weight in [.25, .5, .75]:
            mixture = f'blend_{name}_control{weight}'
            graphs[mixture] = {'lightgbm_control': weight, name: 1-weight}
            pred[mixture] = weight*pred.lightgbm_control + (1-weight)*pred[name]
    graphs['blend_three_families'] = {'lightgbm_control': 1/3, stat: 1/3, gru: 1/3}
    pred['blend_three_families'] = pred[list(graphs['blend_three_families'])].mean(axis=1)
    table = pd.DataFrame([{'model': name, 'average_precision': average_precision_score(y, pred[name])} for name in graphs]).sort_values('average_precision', ascending=False)
    winner = table.iloc[0].model
    base.dump(OUT / 'selection_before_refit.json', {
        'model': winner, 'parts': graphs[winner], 'criterion': 'Oct-Nov 2025 AP',
        'tune_rows': len(pred), 'tune_positives': int(y.sum()), 'family_winners': selected,
        'source_sha256': {str(p.relative_to(base.ROOT)): base.checksum(p) for p in [
            STAT / 'configuration.json', SEQUENCE / 'configuration.json', OUT / 'protocol.md']},
        'december_used': False, 'test_2026_used': False})
    table.to_csv(OUT / 'tuning_mixtures.csv', index=False)
    pred[list(dict.fromkeys(keys + [winner, 'lightgbm_control']))].to_parquet(base.CACHE / 'object_combined_tune.parquet', index=False)
    print(table.to_string(index=False), flush=True)
    print('Selected', winner, graphs[winner], flush=True)


def component_scores(mart, cfg):
    scores = {}
    for name, node in cfg['components'].items():
        if node['kind'] == 'lgb':
            model = lgb.Booster(model_file=str(base.ROOT / node['path']))
            scores[name] = model.predict(mart[node['features']], num_threads=2)
        elif node['kind'] == 'statistical':
            prior = [np.asarray(p) for p in node['priors']]
            scores[name] = statistical_model.statistical_scores(mart, prior)[name]
        elif node['kind'] == 'autoregression':
            model = joblib.load(base.ROOT / node['path'])
            scores[name] = model.predict_proba(statistical_model.autoregressive_frame(mart))[:, 1]
        elif node['kind'] == 'gru':
            from object_sequence_candidate import load_predictor
            scores[name] = load_predictor(base.ROOT / node['path'])(mart.reset_index(drop=True))
        else:
            raise ValueError('Unknown component')
    scores['selected'] = sum(weight * scores[name] for name, weight in cfg['parts'].items())
    return scores


def finalize():
    cfg = json.loads((OUT / 'selection_before_refit.json').read_text())
    for name, expected in cfg['source_sha256'].items():
        if base.checksum(base.ROOT / name) != expected:
            raise ValueError('Selection input changed: '+name)
    mart, dates, _ = base.prepare(end_date='2025-12-31')
    ti = mart.index[mart.date <= '2025-11-28']
    ci = mart.index[mart.date.between('2025-12-01', '2025-12-29')]
    assert dates.loc[ti].max() <= pd.Timestamp('2025-11-30')
    assert dates.loc[ci].max() <= pd.Timestamp('2025-12-31')
    model_dir = base.ROOT / 'models/object_combined'; model_dir.mkdir(parents=True, exist_ok=True)
    cfg['components'] = {}
    old = json.loads((base.OUT / 'configuration.json').read_text())
    params = old['parameters'].copy(); params.update(n_estimators=old['best_iteration'], n_jobs=2)
    control = lgb.LGBMClassifier(**params)
    control.fit(mart.loc[ti, old['features']], mart.loc[ti, base.TARGET], categorical_feature=old['categorical'])
    assert control.booster_.current_iteration() == old['best_iteration']
    path = model_dir / 'lightgbm_control.txt'; control.booster_.save_model(str(path))
    cfg['components']['lightgbm_control'] = {'kind': 'lgb', 'features': old['features'], 'path': str(path.relative_to(base.ROOT)), 'sha256': base.checksum(path)}
    for name in cfg['parts']:
        if name == 'lightgbm_control':
            continue
        if name.startswith(('markov_', 'seasonal_')):
            priors = statistical_model.statistical_priors(mart, pd.Timestamp('2025-11-28'))
            cfg['components'][name] = {'kind': 'statistical', 'priors': [p.tolist() for p in priors]}
        elif name.startswith(('linear_', 'spline_')):
            family, regularization = name.split('_')
            frame = statistical_model.autoregressive_frame(mart)
            model = statistical_model.fit_autoregression(frame.loc[ti], mart.loc[ti, base.TARGET], family, float(regularization))
            path = model_dir / f'{name}.joblib'; joblib.dump(model, path)
            cfg['components'][name] = {'kind': 'autoregression', 'path': str(path.relative_to(base.ROOT)), 'sha256': base.checksum(path)}
        elif name.startswith('gru_'):
            from object_sequence_candidate import refit_fixed_epochs
            source = SEQUENCE / 'models' / f'{name}.pt'; path = model_dir / f'{name}.pt'
            refit_fixed_epochs(mart, source, pd.Timestamp('2025-11-28'), pd.Timestamp('2025-11-30'), path)
            cfg['components'][name] = {'kind': 'gru', 'path': str(path.relative_to(base.ROOT)), 'sha256': base.checksum(path)}
        else:
            raise ValueError('Unsupported selected model')
    cfg.update(refit_feature_end='2025-11-28', refit_label_end='2025-11-30', calibration_feature_start='2025-12-01', calibration_feature_end='2025-12-29')
    base.dump(OUT / 'refit_before_calibration.json', cfg)
    scores = component_scores(mart, cfg)
    cfg['thresholds'] = {name: base.threshold_and_feasibility(mart.loc[ci, base.TARGET].to_numpy(), scores[name][ci])[0] for name in ['selected', 'lightgbm_control']}
    base.dump(OUT / 'selection.json', cfg)
    rows = [base.metrics(name, 'december_calibration', mart.loc[ci, base.TARGET].to_numpy(), scores[name][ci], cfg['thresholds'][name], 29) for name in cfg['thresholds']]
    pd.DataFrame(rows).to_csv(OUT / 'calibration.csv', index=False)
    prediction = mart.loc[ci, ['object_id', 'date', base.TARGET]].copy()
    for name in cfg['thresholds']:
        prediction[name] = scores[name][ci]
    prediction.to_parquet(base.CACHE / 'object_combined_calibration.parquet', index=False)
    print('Frozen before 2026:', cfg['parts'], cfg['thresholds'], flush=True)


def evaluate():
    cfg = json.loads((OUT / 'selection.json').read_text())
    for node in cfg['components'].values():
        if 'path' in node and base.checksum(base.ROOT / node['path']) != node['sha256']:
            raise ValueError('Artifact changed')
    mart, label_dates, coverage = base.prepare(end_date='2026-06-30')
    idx = mart.index[mart.date.between('2025-12-31', '2026-06-28')]
    assert len(idx) == 14040 and mart.loc[idx, 'date'].nunique() == 180
    scores = component_scores(mart, cfg)
    prediction = mart.loc[idx, ['object_id', 'date', 'alarm_today', base.TARGET]].copy()
    for name in cfg['thresholds']:
        prediction[name] = scores[name][idx]
    old = json.loads((base.OUT / 'configuration.json').read_text())
    control = lgb.Booster(model_file=str(base.ROOT / old['model_path']))
    prediction['previous_frozen'] = control.predict(mart.loc[idx, old['features']], num_threads=2)
    expected = pd.read_parquet(base.CACHE / 'object_forward_predictions.parquet')
    # The previous check uses feature_date to make the forecast time explicit.
    date_key = 'feature_date' if 'feature_date' in expected else 'date'
    expected = expected.rename(columns={date_key: 'date'})
    score_key = 'lightgbm_object' if 'lightgbm_object' in expected else 'score'
    matched = prediction[['object_id', 'date', base.TARGET, 'previous_frozen']].merge(expected[['object_id', 'date', base.TARGET, score_key]], on=['object_id', 'date', base.TARGET], validate='one_to_one')
    if len(matched) != len(prediction):
        raise ValueError('Forward check identities differ')
    np.testing.assert_allclose(matched.previous_frozen, matched[score_key], atol=1e-12)
    thresholds = {**cfg['thresholds'], 'previous_frozen': old['thresholds']['lightgbm_object']}
    rows = []; monthly = []; budgets = []
    for name, threshold in thresholds.items():
        for cohort, mask in [('all_object_days', np.ones(len(prediction), dtype=bool)), ('no_alarm_on_feature_day', prediction.alarm_today.to_numpy() == 0)]:
            part = prediction.loc[mask]
            rows.append(base.metrics(name, cohort, part[base.TARGET].to_numpy(), part[name].to_numpy(), threshold, 180))
        for month, part in prediction.groupby((prediction.date + pd.Timedelta(days=1)).dt.to_period('M')):
            monthly.append({'issue_month': str(month), **base.metrics(name, 'all', part[base.TARGET].to_numpy(), part[name].to_numpy(), threshold, part.date.nunique())})
        for budget in [5, 10, 20]:
            ordered = prediction.sort_values(['date', name, 'object_id'], ascending=[True, False, True])
            selected = ordered.groupby('date', observed=True).head(budget)
            tp = int(selected[base.TARGET].sum())
            budgets.append({'model': name, 'objects_per_day': budget, 'true_positive': tp, 'false_positive': len(selected)-tp,
                            'precision': tp/len(selected), 'recall': tp/prediction[base.TARGET].sum(), 'days': 180})
    pd.DataFrame(rows).to_csv(OUT / 'test_metrics.csv', index=False)
    pd.DataFrame(monthly).to_csv(OUT / 'monthly.csv', index=False)
    pd.DataFrame(budgets).to_csv(OUT / 'daily_budget.csv', index=False)
    prediction.to_parquet(base.CACHE / 'object_combined_test.parquet', index=False)
    base.dump(OUT / 'test_check.json', {'rows': len(idx), 'days': 180, 'minimum_lead_hours': 24,
        'first_target_day': str(label_dates.loc[idx].min()), 'last_target_day': str(label_dates.loc[idx].max()),
        'previous_scores_max_difference': float(np.abs(matched.previous_frozen-matched[score_key]).max()),
        'selection_sha256': base.checksum(OUT / 'selection.json'), 'coverage': coverage})
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['select', 'finalize', 'evaluate'])
    args = parser.parse_args(); OUT.mkdir(exist_ok=True, parents=True)
    globals()[args.phase]()
