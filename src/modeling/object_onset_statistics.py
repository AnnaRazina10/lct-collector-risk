"""Fixed statistical candidates for registered onset at feature date D + 2.

No data reading, splitting, training entry point, or evaluation is performed here.
The caller supplies causal daily features and restricts fit() to its training
period. Only fit() reads onset_target. Derived predictors are row-local, so
appending or changing later rows cannot change an earlier predictor row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

TARGET = 'onset_target'
SEQUENCE_COLUMNS = ('alarm_today', 'alarm_channels', 'observed_channels',
                    'fault_channels', 'no_power_channels', 'events_count')
ROW_COLUMNS = ('alarm_today', 'observed_today', 'observed_fraction',
               'alarm_channel_fraction', 'historical_alarm_frequency',
               'alarm_channels', 'observed_channels', 'fault_channels',
               'no_power_channels', 'events_count', 'catalog_channels',
               'alarm_streak', 'days_since_alarm')
HISTORY_COLUMNS = tuple(f'{name}_lag{lag}' for name in SEQUENCE_COLUMNS for lag in (1, 2, 7, 14)) + tuple(
    f'{name}_mean{window}' for name in SEQUENCE_COLUMNS for window in (3, 7, 28))
NUMERIC_INPUTS = ROW_COLUMNS + HISTORY_COLUMNS
CATEGORICAL_COLUMNS = ('object_id', 'object_kind', 'parent_id')
DERIVED_COLUMNS = ('never_registered_alarm', 'target_week_sin', 'target_week_cos',
                   'target_year_sin', 'target_year_cos')


def _labels(frame):
    if TARGET not in frame:
        raise ValueError('fit requires onset_target')
    labels = pd.to_numeric(frame[TARGET], errors='raise').to_numpy(dtype=float)
    if not len(labels) or not np.isfinite(labels).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError('Training onset_target must be nonempty, known, and binary')
    return labels.astype(np.int8)


def _object_ids(frame):
    if 'object_id' not in frame:
        raise ValueError('object_id is required')
    return frame.object_id.astype('string').fillna('__MISSING__').astype(str)


def causal_design(frame):
    """Select an explicit allowlist; never access target or future columns.

    Existing lag/rolling features must already end on D. Calendar predictors
    refer to the known date D+2, not observations made on that future day.
    Counts and recurrence durations get log1p; their scaling/imputation is
    fitted later on training rows only. A separate bit retains the -1
    never-observed-alarm sentinel from days_since_alarm.
    """
    required = {'date', 'object_id', *NUMERIC_INPUTS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError('Missing causal columns: '+', '.join(sorted(missing)))
    dates = pd.to_datetime(frame.date, errors='raise')
    if dates.isna().any() or dates.dt.tz is not None or not dates.eq(dates.dt.normalize()).all():
        raise ValueError('Expected known timezone-free calendar feature dates')
    result = pd.DataFrame(index=frame.index)
    for name in NUMERIC_INPUTS:
        values = pd.to_numeric(frame[name], errors='raise').astype(float)
        if np.isinf(values.to_numpy()).any():
            raise ValueError('Infinite causal feature: '+name)
        # Alarm histories/fractions remain on their original [0,1] scale.
        is_count = name in ('catalog_channels', 'alarm_streak', 'days_since_alarm') or any(
            name == source or name.startswith(source+'_') for source in SEQUENCE_COLUMNS[1:])
        if is_count:
            values = np.log1p(values.clip(lower=0))
        result[name] = values
    result['never_registered_alarm'] = frame.days_since_alarm.eq(-1).astype(float)
    target_date = dates + pd.Timedelta(days=2)
    weekday = target_date.dt.dayofweek
    result['target_week_sin'] = np.sin(2*np.pi*weekday/7)
    result['target_week_cos'] = np.cos(2*np.pi*weekday/7)
    result['target_year_sin'] = np.sin(2*np.pi*(target_date.dt.dayofyear-1)/365.25)
    result['target_year_cos'] = np.cos(2*np.pi*(target_date.dt.dayofyear-1)/365.25)
    for name in CATEGORICAL_COLUMNS:
        if name == 'object_id':
            result[name] = _object_ids(frame)
        elif name in frame:
            result[name] = frame[name].astype('string').fillna('__MISSING__').astype(str)
        else:
            result[name] = '__MISSING__'
    return result


class OnsetLogistic(BaseEstimator):
    """One predetermined C=1 L2 logistic model, without balanced reweighting.

    API: OnsetLogistic().fit(train_frame).predict(frame) -> positive scores.
    Complete instances can be persisted with pickle/joblib. Unknown objects
    receive the learned non-identity contributions (one-hot ignore behavior).
    """

    def fit(self, frame):
        labels = _labels(frame)
        if np.unique(labels).size != 2:
            raise ValueError('Logistic fit requires both onset classes')
        design = causal_design(frame)
        numeric = list(NUMERIC_INPUTS+DERIVED_COLUMNS)
        preprocessing = ColumnTransformer([
            ('numeric', Pipeline([
                ('imputer', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
                ('scaler', StandardScaler()),
            ]), numeric),
            ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), list(CATEGORICAL_COLUMNS)),
        ], remainder='drop', sparse_threshold=0)
        self.pipeline_ = Pipeline([
            ('preprocess', preprocessing),
            ('logistic', LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000,
                                            tol=1e-5, random_state=84)),
        ])
        with threadpool_limits(limits=2):
            self.pipeline_.fit(design, labels)
        self.classes_ = np.array([0, 1])
        self.training_rows_ = len(frame)
        self.training_positive_rate_ = float(labels.mean())
        self.training_feature_end_ = str(pd.to_datetime(frame.date).max().date())
        self.n_iter_ = int(self.pipeline_.named_steps['logistic'].n_iter_[0])
        return self

    def predict(self, frame):
        check_is_fitted(self, 'pipeline_')
        design = causal_design(frame)
        if frame.empty:
            return np.empty(0, dtype=float)
        with threadpool_limits(limits=2):
            return self.pipeline_.predict_proba(design)[:, 1]


class SmoothedOnsetFrequency(BaseEstimator):
    """Frozen object onset frequency with 30 pseudo-observations by default.

    The Beta prior is centered on the global training onset rate. predict()
    does not update counts, so tune/test labels never modify the estimate.
    Unknown objects receive that global training rate.
    """

    def __init__(self, prior_strength=30.0):
        self.prior_strength = prior_strength

    def fit(self, frame):
        if not np.isfinite(self.prior_strength) or self.prior_strength <= 0:
            raise ValueError('prior_strength must be finite and positive')
        labels = _labels(frame)
        ids = _object_ids(frame).to_numpy()
        self.global_rate_ = float(labels.mean())
        aggregates = pd.DataFrame({'object_id': ids, 'label': labels}).groupby('object_id').label.agg(['sum', 'count'])
        values = (aggregates['sum']+self.prior_strength*self.global_rate_)/(aggregates['count']+self.prior_strength)
        self.rates_ = values.to_dict()
        self.training_rows_ = len(frame)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, frame):
        check_is_fitted(self, 'rates_')
        return _object_ids(frame).map(self.rates_).fillna(self.global_rate_).to_numpy(dtype=float)
