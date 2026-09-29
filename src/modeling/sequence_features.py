"""Causal per-channel event memory. Never reads next-day labels."""
from __future__ import annotations
import numpy as np
import pandas as pd
from improve_baseline import rolling_sum


def carry_last(values, known):
    """Forward carry with explicit age; missing history stays unknown."""
    n, d = values.shape
    times = np.arange(d)[None, :]
    positions = np.maximum.accumulate(np.where(known, times, -1), axis=1)
    carried = np.take_along_axis(values, np.maximum(positions, 0), axis=1)
    return np.where(positions >= 0, carried, np.nan), np.where(positions >= 0, times-positions, np.nan)


def sequence_probabilities(alarm, observed, order):
    """Update yesterday's context using today's known outcome, then query today.

    Each channel has its own state table. Exponential discount prevents permanent
    dominance of old regimes. Smoothing uses only the channel's observed past.
    """
    n, d = alarm.shape
    size = 4 ** order
    exposure = np.zeros((n, size), np.float32)
    yes = np.zeros_like(exposure)
    active = np.zeros_like(exposure)
    context = np.zeros(n, np.int32)
    prior_yes = np.zeros(n, np.float32)
    prior_obs = np.zeros(n, np.float32)
    prior_n = 0.
    rows = np.arange(n)
    pa, po, support = (np.empty((n, d), np.float32) for _ in range(3))
    decay = .5 ** (1 / 90)
    for t in range(d):
        exposure *= decay; yes *= decay; active *= decay
        if t >= order:
            exposure[rows, context] += 1
            yes[rows, context] += alarm[:, t]
            active[rows, context] += observed[:, t]
        prior_yes = prior_yes * decay + alarm[:, t]
        prior_obs = prior_obs * decay + observed[:, t]
        prior_n = prior_n * decay + 1
        context = ((context * 4) + 2 * alarm[:, t] + observed[:, t]) % size
        s = exposure[rows, context]
        p = (prior_yes + .1) / (prior_n + 10)
        o = (prior_obs + 1) / (prior_n + 10)
        pa[:, t] = (yes[rows, context] + 10*p) / (s + 10)
        po[:, t] = (active[rows, context] + 10*o) / (s + 10)
        support[:, t] = s
    return pa, po, support


def recurrence(mask):
    n, d = mask.shape
    last = np.full(n, -1)
    mean_gap = np.full(n, np.nan, np.float32)
    previous_gap = np.full(n, np.nan, np.float32)
    gap_sum = np.zeros(n)
    gap_n = np.zeros(n)
    age, gap, ema, cv = (np.empty((n, d), np.float32) for _ in range(4))
    gap_sq = np.zeros(n)
    for t in range(d):
        update = mask[:, t] & (last >= 0)
        interval = t - last[update]
        gap_sum[update] += interval
        gap_sq[update] += interval**2
        gap_n[update] += 1
        previous_gap[update] = interval
        mean_gap[update] = np.where(np.isnan(mean_gap[update]), interval,
                                   .3*interval + .7*mean_gap[update])
        last[mask[:, t]] = t
        age[:, t] = np.where(last >= 0, t-last, np.nan)
        gap[:, t] = previous_gap
        ema[:, t] = mean_gap
        avg = gap_sum / np.maximum(gap_n, 1)
        sd = np.sqrt(np.maximum(gap_sq/np.maximum(gap_n, 1)-avg**2, 0))
        cv[:, t] = np.where(gap_n > 1, sd/(avg+1e-8), np.nan)
    return age, gap, ema, cv


def build_sequence_features(mart):
    keys = mart[['channel_id', 'date']]
    if not keys.equals(keys.sort_values(['channel_id', 'date'])) or keys.duplicated().any():
        raise ValueError('Expected unique channel-sorted grid')
    counts = mart.groupby('channel_id', sort=False).size()
    if counts.nunique() != 1:
        raise ValueError('Expected complete daily grid')
    n, d = len(counts), int(counts.iloc[0])
    dates = pd.DatetimeIndex(mart.date.iloc[:d])
    if not np.array_equal(np.diff(dates.values).astype('timedelta64[D]').astype(int), np.ones(d-1)):
        raise ValueError('Dates must be consecutive')
    if not np.array_equal(mart.date.to_numpy().reshape(n,d), np.broadcast_to(dates.values,(n,d))):
        raise ValueError('Channel grids differ')
    def arr(c): return mart[c].to_numpy().reshape(n,d)
    new = {}
    def add(c, a): new['seq_'+c] = np.asarray(a, np.float32).ravel()
    alarm = (arr('alarm_count') > 0).astype(np.int32)
    observed = (arr('events_count') > 0).astype(np.int32)
    for order in (1, 2, 3):
        pa, po, count = sequence_probabilities(alarm, observed, order)
        add(f'pattern{order}_alarm', pa)
        add(f'pattern{order}_observed', po)
        add(f'pattern{order}_support', count)
    for name, mask in [('alarm', alarm), ('observed', observed)]:
        age, gap, ema, cv = recurrence(mask.astype(bool))
        add(name+'_last_gap', gap); add(name+'_gap_ema', ema)
        add(name+'_gap_cv', cv); add(name+'_cycle_fraction', (age+1)/(ema+1))
        # Tomorrow's weekday corresponds to t-6, t-13, ... (not t-7).
        for weeks in (4, 12):
            total = np.zeros_like(alarm, np.float32)
            exposure = np.zeros(d, np.float32)
            for k in range(weeks):
                lag = 6 + 7*k
                if lag >= d: break
                total[:, lag:] += mask[:, :-lag]
                exposure[lag:] += 1
            add(name+f'_tomorrow_weekday_{weeks}w', (total+.1)/(exposure[None,:]+10))
        for window in (2, 5, 14, 60, 180):
            add(name+f'_rate_{window}d', rolling_sum(mask,window)/np.minimum(np.arange(d)+1,window)[None,:])
    # Last known state is a memory feature, NOT a claim that a missing day is healthy.
    known = observed.astype(bool)
    for col in ['intra_last_event_is_alarm','intra_last_event_minute','intra_last_alarm_minute',
                'value_mean_1d','value_std_1d','events_count','alarm_count','fault_count','no_power_count']:
        values = arr(col).astype(np.float32)
        valid = known & np.isfinite(values)
        if col.startswith('value_'): valid &= arr('numeric_count') > 0
        carried, age = carry_last(values, valid)
        add('last_known_'+col,carried)
        if col in ['intra_last_event_is_alarm','value_mean_1d']: add('age_'+col,age)
    last_minute, age = carry_last(arr('intra_last_event_minute'), known)
    add('minutes_since_event', (age+1)*1440-last_minute)
    last_alarm_min, age_alarm = carry_last(arr('intra_last_alarm_minute'), arr('intra_last_alarm_minute')>=0)
    add('minutes_since_alarm', (age_alarm+1)*1440-last_alarm_min)
    for col in ['fault_count','no_power_count','events_count']:
        values = np.log1p(arr(col))
        for days in (1, 2, 6):
            out = np.full_like(values, np.nan, dtype=np.float32)
            if days < d: out[:, days:] = values[:, :-days]
            add(f'{col}_lag{days}', out)
    return pd.concat([keys.reset_index(drop=True),pd.DataFrame(new)],axis=1)
