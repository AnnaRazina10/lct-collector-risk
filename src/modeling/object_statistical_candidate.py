"""Causal statistical and autoregressive alternatives for an object D+2 target."""
from __future__ import annotations
import json
import time
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
import object_risk_probe as base

ROOT = base.ROOT
OUT = ROOT / 'reports/object_statistical_candidate'
MODELS = ROOT / 'models/object_statistical'


def markov_forecast(states, prior, half_life=90, strength=5):
    """Update from known transition into D, then propagate twice to D+2."""
    counts = np.zeros((3, 3), dtype=float)
    scores = np.empty(len(states))
    decay = 2 ** (-1 / half_life)
    for i, state in enumerate(states):
        counts *= decay
        if i:
            counts[states[i-1], state] += 1
        transition = (counts + strength * prior) / (counts.sum(axis=1, keepdims=True) + strength)
        scores[i] = (transition @ transition)[state, 2]
    return scores


def seasonal_forecast(alarm, weekdays, prior, half_life=90, strength=10):
    counts = np.zeros(7); positives = np.zeros(7); scores = np.empty(len(alarm))
    decay = 2 ** (-1 / half_life)
    for i, (value, weekday) in enumerate(zip(alarm, weekdays)):
        counts *= decay; positives *= decay
        counts[weekday] += 1; positives[weekday] += value
        target_weekday = (weekday + 2) % 7
        scores[i] = (positives[target_weekday] + strength * prior[target_weekday]) / (counts[target_weekday] + strength)
    return scores


def statistical_priors(mart, cutoff):
    train = mart.loc[mart.date <= cutoff].copy()
    train['state'] = np.where(train.alarm_today > 0, 2, np.where(train.observed_today > 0, 1, 0))
    transition = np.ones((3, 3), dtype=float)
    for _, group in train.groupby('object_id', observed=True, sort=False):
        states = group.state.to_numpy()
        np.add.at(transition, (states[:-1], states[1:]), 1)
    transition /= transition.sum(axis=1, keepdims=True)
    weekday = train.groupby(train.date.dt.dayofweek).alarm_today.agg(['sum', 'count'])
    seasonal = ((weekday['sum'] + 1) / (weekday['count'] + 2)).reindex(range(7)).to_numpy()
    return transition, seasonal


def statistical_scores(mart, priors):
    scores = {name: np.empty(len(mart)) for name in ['markov_45', 'markov_180', 'seasonal_90', 'seasonal_365']}
    for _, group in mart.groupby('object_id', observed=True, sort=False):
        if not group.date.diff().dropna().eq(pd.Timedelta(days=1)).all():
            raise ValueError('Expected a complete, ordered daily calendar per object')
        positions = mart.index.get_indexer(group.index)
        state = np.where(group.alarm_today > 0, 2, np.where(group.observed_today > 0, 1, 0))
        for half in [45, 180]:
            scores[f'markov_{half}'][positions] = markov_forecast(state, priors[0], half)
        for half in [90, 365]:
            scores[f'seasonal_{half}'][positions] = seasonal_forecast(group.alarm_today.to_numpy(), group.date.dt.dayofweek.to_numpy(), priors[1], half)
    return scores


def autoregressive_frame(mart):
    cols = [c for c in mart if c not in [base.TARGET, 'date', 'object_kind', 'parent_id']]
    frame = mart[cols].copy()
    for col in cols:
        if col == 'object_id':
            frame[col] = frame[col].astype(str)
        else:
            frame[col] = frame[col].astype(float).replace([np.inf, -np.inf], np.nan)
            if col == 'days_since_alarm':
                frame[col] = frame[col].where(frame[col] >= 0)
            if col not in ['annual_sin', 'annual_cos', 'target_dayofweek']:
                frame[col] = np.sign(frame[col]) * np.log1p(np.abs(frame[col]))
    weekday = frame.pop('target_dayofweek')
    frame['weekday_sin'] = np.sin(2*np.pi*weekday/7)
    frame['weekday_cos'] = np.cos(2*np.pi*weekday/7)
    return frame


def fit_autoregression(frame, y, family, regularization):
    numeric = [c for c in frame if c != 'object_id']
    transforms = [SimpleImputer(strategy='median', add_indicator=True)]
    if family == 'spline':
        transforms.append(SplineTransformer(n_knots=4, degree=2, knots='quantile', extrapolation='constant', include_bias=False))
    transforms.append(StandardScaler())
    prep = ColumnTransformer([
        ('numeric', make_pipeline(*transforms), numeric),
        ('object', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['object_id'])])
    model = make_pipeline(prep, LogisticRegression(C=regularization, max_iter=1000, solver='lbfgs'))
    model.fit(frame, y)
    return model


def main():
    OUT.mkdir(exist_ok=True, parents=True); MODELS.mkdir(exist_ok=True, parents=True)
    started = time.time()
    mart, label_dates, coverage = base.prepare()
    ti = mart.index[mart.date <= '2025-09-28']
    vi = mart.index[mart.date.between('2025-10-01', '2025-11-28')]
    assert label_dates.loc[ti].max() <= pd.Timestamp('2025-09-30')
    assert label_dates.loc[vi].max() <= pd.Timestamp('2025-11-30')
    y = mart.loc[vi, base.TARGET].to_numpy()
    priors = statistical_priors(mart, pd.Timestamp('2025-09-28'))
    all_scores = statistical_scores(mart, priors)
    scores = {name: values[vi] for name, values in all_scores.items()}
    config = {'priors': [p.tolist() for p in priors], 'models': {}, 'coverage': coverage}
    frame = autoregressive_frame(mart)
    for family in ['linear', 'spline']:
        for c in [.1, 1.0]:
            name = f'{family}_{c}'
            model = fit_autoregression(frame.loc[ti], mart.loc[ti, base.TARGET], family, c)
            path = MODELS / f'{name}.joblib'
            joblib.dump(model, path)
            scores[name] = model.predict_proba(frame.loc[vi])[:, 1]
            restored = joblib.load(path).predict_proba(frame.loc[vi[:256]])[:, 1]
            np.testing.assert_allclose(restored, scores[name][:256], atol=1e-12)
            config['models'][name] = {'family': family, 'C': c, 'features': list(frame), 'artifact': str(path.relative_to(ROOT)),
                                      'sha256': base.checksum(path), 'reload_max_difference': float(np.abs(restored-scores[name][:256]).max())}
            print('Completed', name, flush=True)
    old = json.loads((base.OUT / 'configuration.json').read_text())
    model = lgb.Booster(model_file=str(ROOT / old['model_path']))
    if base.checksum(ROOT / old['model_path']) != old['model_sha256']:
        raise ValueError('Control artifact changed')
    scores['lightgbm_control'] = model.predict(mart.loc[vi, old['features']], num_threads=2)
    historical = pd.read_parquet(base.CACHE / 'object_risk_tuning_predictions.parquet')
    reproduced = mart.loc[vi, ['object_id', 'date']].assign(score=scores['lightgbm_control']).merge(historical, on=['object_id', 'date'], validate='one_to_one')
    np.testing.assert_allclose(reproduced.score, reproduced.lightgbm_object, atol=1e-12)
    frequency = mart.loc[ti].groupby('object_id', observed=True)[base.TARGET].mean()
    scores['historical_frequency'] = mart.loc[vi, 'object_id'].map(frequency).astype(float).to_numpy()
    scores['persistence'] = mart.loc[vi, 'alarm_today'].to_numpy()
    rows = []
    predictions = mart.loc[vi, ['object_id', 'date', base.TARGET]].copy()
    for name, score in scores.items():
        threshold, _ = base.threshold_and_feasibility(y, score)
        rows.append(base.metrics(name, 'tune_only', y, score, threshold, mart.loc[vi, 'date'].nunique()))
        predictions[name] = score
    table = pd.DataFrame(rows).sort_values('average_precision', ascending=False)
    table.to_csv(OUT / 'tuning.csv', index=False)
    predictions.to_parquet(base.CACHE / 'object_statistical_tuning.parquet', index=False)
    config.update(train_end='2025-09-28', tune_start='2025-10-01', tune_end='2025-11-28',
                  maximum_label_date=str(label_dates.max()), december_used=False, test_2026_used=False,
                  training_rows=len(ti), tuning_rows=len(vi), seconds=time.time()-started,
                  code_sha256=base.checksum(Path(__file__)), protocol_sha256=base.checksum(OUT / 'protocol.md'),
                  selected_statistical=table.loc[~table.model.isin(['lightgbm_control', 'historical_frequency', 'persistence'])].iloc[0].model,
                  control_reload_max_difference=float(np.abs(reproduced.score-reproduced.lightgbm_object).max()))
    base.dump(OUT / 'configuration.json', config)
    print(table[['model', 'average_precision', 'precision', 'recall', 'f1']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
