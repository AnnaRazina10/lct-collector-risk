"""Paired temporal-block uncertainty for fixed predictions, never model selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


class WeightedAP:
    """Cache score ordering and tied groups for repeated weighted AP calculations."""

    def __init__(self, y, score, day):
        order = np.argsort(-np.asarray(score), kind='stable')
        ordered_score = np.asarray(score)[order]
        self.y = np.asarray(y)[order]
        self.day = np.asarray(day)[order]
        self.ends = np.r_[np.flatnonzero(np.diff(ordered_score)), len(order) - 1]

    def __call__(self, day_weights):
        weights = np.asarray(day_weights)[self.day]
        tp = np.cumsum(weights * self.y)[self.ends]
        total = np.cumsum(weights)[self.ends]
        if not tp[-1]:
            return 0.0
        precision = np.divide(tp, total, out=np.zeros(len(tp), dtype=float), where=total > 0)
        return float(np.dot(np.diff(np.r_[0, tp]), precision) / tp[-1])


def main():
    out = ROOT / 'reports/improved_v3'
    static = pd.read_parquet(out / 'predictions/test.parquet')
    rolling = pd.read_parquet(ROOT / 'reports/rolling_update/predictions/test.parquet')
    keys = ['channel_id', 'date', 'target_alarm_next_24h']
    data = static.merge(rolling[keys + ['rolling_lgb', 'rolling_blend']], on=keys, validate='one_to_one')
    assert len(data) == 1200000
    days, unique = pd.factorize(data.date, sort=True)
    assert (pd.Series(unique).diff().dropna() == pd.Timedelta(days=1)).all()
    names = ['score_v2', 'score_v3', 'rolling_lgb', 'rolling_blend']
    metrics = {name: WeightedAP(data[keys[-1]], data[name], days) for name in names}
    rng = np.random.default_rng(139)
    draws = {name: [] for name in names[1:]}
    for _ in range(200):
        starts = rng.integers(0, len(unique), size=int(np.ceil(len(unique) / 7)))
        sampled = ((starts[:, None] + np.arange(7)) % len(unique)).ravel()[:len(unique)]
        weights = np.bincount(sampled, minlength=len(unique))
        baseline = metrics['score_v2'](weights)
        for name in draws:
            draws[name].append(metrics[name](weights) - baseline)
    result = {
        'method': '200 paired circular moving-block bootstrap draws; 7 calendar days per block',
        'seed': 139, 'rows': len(data), 'days': len(unique),
        'interpretation': 'Descriptive uncertainty conditional on frozen scores; no retraining, no model selection; rolling models use earlier 2026 labels',
        'AP_difference_vs_v2': {name: {'lower_2_5': float(np.quantile(values, .025)),
                                      'upper_97_5': float(np.quantile(values, .975))}
                                for name, values in draws.items()}}
    (out / 'uncertainty.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
