"""Fixed-budget evaluation of registered alarm onsets, with paired week blocks.

``date`` is the feature calendar day D (midnight, Moscow calendar), the issue day
is D+1, and labels refer to the target window D+2. Scores are positional arrays in
the original row order. Labels are validated but never used to select warnings.
No threshold or k is fitted here. Call sensitivity k=5/20 separately if planned.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

REQUIRED = ("object_id", "date", "onset_target", "alarm_target")
FEW_WEEK_THRESHOLD = 8  # Reporting heuristic; not an inferential guarantee.


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validated(frame, scores, k):
    k = _positive_integer(k, "k")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("frame must be a nonempty DataFrame")
    missing = set(REQUIRED).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing evaluation columns: {sorted(missing)}")
    if frame[list(REQUIRED)].isna().any().any():
        raise ValueError("Identifiers, dates and labels must be explicit, without missing values")
    identifiers = frame.object_id.astype(str).to_numpy()
    if np.any(np.char.strip(identifiers.astype(str)) == ""):
        raise ValueError("object_id cannot be blank")
    try:
        days = pd.to_datetime(frame.date, errors="raise")
        if isinstance(days.dtype, pd.DatetimeTZDtype):
            days = days.dt.tz_convert("Europe/Moscow").dt.tz_localize(None)
        if days.isna().any() or not days.eq(days.dt.normalize()).all():
            raise ValueError("date must identify a feature calendar day at midnight")
        dates = days.to_numpy(dtype="datetime64[ns]")
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError("Invalid feature calendar dates") from error
    if pd.DataFrame({"object_id": identifiers, "date": dates}).duplicated().any():
        raise ValueError("Duplicate object_id/date after string identifier normalization")
    labels = []
    for name in ("onset_target", "alarm_target"):
        if not frame[name].isin([0, 1]).all():
            raise ValueError(f"{name} must contain only explicit binary 0/1 values")
        labels.append(frame[name].to_numpy(dtype=np.int8))
    onsets, alarms = labels
    if np.any(onsets > alarms):
        raise ValueError("An onset must also be an alarm: onset_target <= alarm_target")
    try:
        values = np.asarray(scores, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("scores must be a finite numeric array") from error
    if values.ndim != 1 or len(values) != len(frame) or not np.isfinite(values).all():
        raise ValueError("scores must be a finite 1D array matching frame rows")
    unique_days, day_counts = np.unique(dates, return_counts=True)
    if np.any(day_counts < k):
        raise ValueError("Every day must have at least k candidates; equal daily budget cannot be silently reduced")
    return identifiers, dates, onsets, alarms, values, unique_days, k


def _select(identifiers, dates, scores, k):
    # lexsort's final key is primary. String ids resolve exact score ties.
    order = np.lexsort((identifiers, -scores, dates))
    sorted_dates = dates[order]
    starts = np.flatnonzero(np.r_[True, sorted_dates[1:] != sorted_dates[:-1]])
    counts = np.diff(np.r_[starts, len(order)])
    rank_within_day = np.arange(len(order)) - np.repeat(starts, counts)
    chosen = np.zeros(len(order), dtype=bool)
    chosen[order[rank_within_day < k]] = True
    return chosen


def daily_topk(frame, scores, k=10) -> np.ndarray:
    """Select exactly k warnings per day; return bools in the input row order.

    Unequal candidate counts are allowed only if every day has at least k rows.
    All four REQUIRED columns are validated; missing/ambiguous labels fail closed.
    """
    identifiers, dates, _, _, values, _, k = _validated(frame, scores, k)
    return _select(identifiers, dates, values, k)


def _summary(identifiers, dates, onsets, alarms, scores, unique_days, k, chosen):
    positives = int(onsets.sum())
    warnings = int(chosen.sum())
    true_positives = int(np.sum(chosen & (onsets == 1)))
    partition = {"onset": true_positives,
                 "repeat_alarm": int(np.sum(chosen & (alarms == 1) & (onsets == 0))),
                 "no_alarm": int(np.sum(chosen & (alarms == 0)))}
    assert sum(partition.values()) == warnings
    return {"rows": len(scores), "objects": len(np.unique(identifiers)), "days": len(unique_days), "k": k,
            "feature_start": str(unique_days[0].astype("datetime64[D]")),
            "feature_end": str(unique_days[-1].astype("datetime64[D]")),
            "issue_start": str((unique_days[0] + np.timedelta64(1, "D")).astype("datetime64[D]")),
            "issue_end": str((unique_days[-1] + np.timedelta64(1, "D")).astype("datetime64[D]")),
            "onsets": positives, "onset_prevalence": positives / len(scores),
            "average_precision": float(average_precision_score(onsets, scores)) if positives else 0.0,
            "precision_at_k": true_positives / warnings,
            "recall": true_positives / positives if positives else None,
            "tp": true_positives, "warnings": warnings, "warnings_per_day": warnings / len(unique_days),
            "warning_partition": partition,
            "undefined_recall": positives == 0,
            "date_semantics": "feature_day_D; issue_day_D_plus_1; target_day_D_plus_2"}


def summarize(frame, scores, k=10) -> dict:
    """Pooled AP and fixed daily top-k P/R for onsets, including every input row.

    AP is 0 by convention if there are no onsets; recall is then None. A warning
    on a repeated alarm is not an onset hit, but is not a false any-alarm warning.
    """
    inputs = _validated(frame, scores, k)
    chosen = _select(inputs[0], inputs[1], inputs[4], inputs[6])
    return _summary(*inputs, chosen)


def paired_week_bootstrap(frame, new_scores, baseline_scores, k=10, n=1000, seed=84) -> dict:
    """Paired percentile intervals for NEW-minus-BASELINE P@k and onset recall.

    Sample observed issue-calendar weeks (Monday–Sunday) with replacement. Every
    block contains all its rows, including partial boundary weeks or calendar
    gaps. No dates are removed. Daily selections are computed before resampling;
    TP/warnings/onsets are summed over sampled blocks before ratios are formed.
    Repeated sampled weeks therefore do not rerank or deduplicate any warnings.

    These are descriptive block intervals, not evidence from an independent new
    test. With one block the reported intervals are None (resampling cannot
    estimate uncertainty). Replicates without onsets have no defined recall.
    """
    n = _positive_integer(n, "n")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    new_inputs = _validated(frame, new_scores, k)
    baseline_inputs = _validated(frame, baseline_scores, k)
    identifiers, dates, onsets, alarms, new_values, unique_days, k = new_inputs
    baseline_values = baseline_inputs[4]
    new_chosen = _select(identifiers, dates, new_values, k)
    baseline_chosen = _select(identifiers, dates, baseline_values, k)
    new_summary = _summary(*new_inputs, new_chosen)
    baseline_summary = _summary(*baseline_inputs, baseline_chosen)

    issue_days = pd.DatetimeIndex(dates) + pd.Timedelta(days=1)
    week_starts = (issue_days-pd.to_timedelta(issue_days.dayofweek, unit="D")).to_numpy()
    weeks, week_index = np.unique(week_starts, return_inverse=True)
    week_count = len(weeks)
    aggregate = lambda weights: np.bincount(week_index, weights=weights, minlength=week_count).astype(np.int64)
    weekly_new_tp = aggregate(new_chosen & (onsets == 1))
    weekly_baseline_tp = aggregate(baseline_chosen & (onsets == 1))
    weekly_warnings = aggregate(new_chosen)
    if not np.array_equal(weekly_warnings, aggregate(baseline_chosen)):
        raise AssertionError("Paired models must use the same daily warning budget")
    weekly_onsets = aggregate(onsets)
    day_week = pd.DataFrame({"week": week_starts, "date": dates}).drop_duplicates()
    day_counts = day_week.groupby("week", sort=True).size().to_numpy(dtype=int)

    rng = np.random.default_rng(int(seed))
    sampled = rng.integers(0, week_count, size=(n, week_count))
    tp_difference = (weekly_new_tp-weekly_baseline_tp)[sampled].sum(axis=1)
    warning_denominator = weekly_warnings[sampled].sum(axis=1)
    onset_denominator = weekly_onsets[sampled].sum(axis=1)
    precision_differences = tp_difference / warning_denominator
    recall_differences = np.full(n, np.nan)
    np.divide(tp_difference, onset_denominator, out=recall_differences, where=onset_denominator > 0)

    def interval(samples):
        valid = samples[np.isfinite(samples)]
        if week_count < 2 or not len(valid):
            return None
        return [float(value) for value in np.quantile(valid, [.025, .975])]

    limitations = ["Descriptive paired calendar-week bootstrap; no independent test is created by resampling.",
                   "Weeks are treated as exchangeable blocks; dependence across weeks and dataset shift are not covered."]
    if week_count < FEW_WEEK_THRESHOLD:
        limitations.append(f"Only {week_count} issue-week blocks (<{FEW_WEEK_THRESHOLD}); intervals may be unstable and should not establish a reliable improvement alone.")
    if week_count == 1:
        limitations.append("One observed week gives no estimable block uncertainty; confidence intervals are omitted.")
    no_onset_replicates = int(np.sum(onset_denominator == 0))
    if no_onset_replicates:
        limitations.append("Recall intervals use only resamples with at least one onset; the count of excluded recall replicates is reported.")
    return {"k": k, "n": n, "seed": int(seed), "confidence_level": .95,
            "difference_direction": "new_minus_baseline", "block_unit": "issue_calendar_week_monday_sunday",
            "date_semantics": "feature_day_D; issue_day_D_plus_1; target_day_D_plus_2",
            "rows": len(frame), "days": len(unique_days), "weeks": week_count,
            "partial_week_blocks": int(np.sum(day_counts < 7)),
            "min_days_per_block": int(day_counts.min()), "max_days_per_block": int(day_counts.max()),
            "all_input_rows_included": True, "few_weeks": week_count < FEW_WEEK_THRESHOLD,
            "few_weeks_threshold": FEW_WEEK_THRESHOLD,
            "week_blocks": [{"issue_week_start": str(week.astype("datetime64[D]")), "days": int(day_counts[i]),
                             "warnings": int(weekly_warnings[i]), "onsets": int(weekly_onsets[i])}
                            for i, week in enumerate(weeks)],
            "new": new_summary, "baseline": baseline_summary,
            "delta_precision_at_k": {"estimate": new_summary["precision_at_k"]-baseline_summary["precision_at_k"],
                                     "ci95": interval(precision_differences)},
            "delta_recall": {"estimate": (new_summary["recall"]-baseline_summary["recall"]) if new_summary["recall"] is not None else None,
                             "ci95": interval(recall_differences)},
            "recall_replicates_without_onsets": no_onset_replicates,
            "valid_recall_replicates": n-no_onset_replicates,
            "limitations": limitations}
