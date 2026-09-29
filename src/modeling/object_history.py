"""Portable, causal object-day history with a caller-frozen object population.

Feature formulas deliberately match ``object_risk_probe.prepare_frames``. The
caller supplies the calendar origin, end, metadata snapshot and eligible object
IDs selected without evaluation outcomes. No files or model weights are read.
Missing daily rows mean no registered records, never confirmed equipment health.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

TARGET = "target_any_object_alarm_d_plus_2"
COUNTS = ["events_count", "alarm_count", "fault_count", "no_power_count", "undefined_count",
          "numeric_count", "text_count", "off_count", "on_count", "movement_detected_count", "open_count"]


def _calendar_boundary(value, name):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is not None or stamp != stamp.normalize():
        raise ValueError(f"{name} must be a known timezone-naive calendar midnight")
    return stamp


def prepare_frames(raw: pd.DataFrame, meta: pd.DataFrame, objects: pd.DataFrame, *,
                   start_date, end_date, eligible_object_ids: Iterable[str]):
    """Return ``(mart, label_date, coverage)`` using the frozen 71 feature formulas.

    ``raw`` contains unique channel-day records and COUNTS within the inclusive
    [start_date, end_date] interval. Dates must already represent local calendar
    midnights. All eligible IDs must exist in both supplied catalogs and have at
    least one catalog channel. Every chosen object receives every calendar day,
    including objects without records. Known but unchosen objects are excluded,
    never added because they happen to appear later.

    Metadata availability and population selection are caller responsibilities:
    pass the same train-cutoff population and admissible metadata to every
    temporal comparison. Features include the current day D; the any-alarm label
    is D+2. The final two labels and their dates are unavailable (NaN/NaT).
    """
    start_date = _calendar_boundary(start_date, "start_date")
    end_date = _calendar_boundary(end_date, "end_date")
    if start_date > end_date:
        raise ValueError("start_date must not exceed end_date")
    required = [(raw, {"channel_id", "date", *COUNTS}, "raw"),
                (meta, {"channel_id", "object_id"}, "meta"),
                (objects, {"object_id", "object_kind", "parent_id"}, "objects")]
    for frame, columns, name in required:
        missing = columns.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if not pd.api.types.is_datetime64_any_dtype(raw.date.dtype):
        raise ValueError("Daily dates must have a datetime64 dtype")
    if raw.date.dt.tz is not None or raw.date.isna().any():
        raise ValueError("Daily frame contains unknown or timezone-aware dates")
    if raw.date.ne(raw.date.dt.normalize()).any():
        raise ValueError("Daily frame contains non-midnight dates")
    if raw.date.lt(start_date).any() or raw.date.gt(end_date).any():
        raise ValueError("Daily frame contains dates outside the requested calendar")
    if raw.channel_id.isna().any() or raw.duplicated(["channel_id", "date"]).any():
        raise ValueError("Unknown channel identifiers or duplicate channel-day records")
    if (meta.channel_id.isna().any() or meta.channel_id.duplicated().any()
            or objects.object_id.isna().any() or objects.object_id.duplicated().any()):
        raise ValueError("Ambiguous metadata identifiers")
    identities = list(eligible_object_ids)
    if not identities or any(not isinstance(value, str) for value in identities):
        raise ValueError("eligible_object_ids must be a nonempty collection of strings")
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate eligible object identifiers")
    identities = sorted(identities)
    eligible_meta = meta[meta.object_id.isin(objects.object_id)]
    known = set(eligible_meta.object_id.unique())
    absent = set(identities).difference(known)
    if absent:
        raise ValueError(f"Eligible objects lack catalog channels: {sorted(absent)}")
    joined = raw.merge(eligible_meta[["channel_id", "object_id"]], on="channel_id", how="left", validate="many_to_one")
    unknown = joined.object_id.isna()
    outside_population = ~unknown & ~joined.object_id.isin(identities)
    included = ~unknown & ~outside_population
    coverage = {"catalog_objects_with_channels": len(identities),
                "all_catalog_objects_with_channels": len(known),
                "excluded_unknown_channel_days": int(unknown.sum()),
                "excluded_alarm_channel_days": int(joined.loc[unknown, "alarm_count"].gt(0).sum()),
                "excluded_noneligible_channel_days": int(outside_population.sum()),
                "excluded_noneligible_alarm_channel_days": int(joined.loc[outside_population, "alarm_count"].gt(0).sum()),
                "included_channel_days": int(included.sum()),
                "start_date": str(start_date.date()), "end_date": str(end_date.date()),
                "eligible_object_ids": identities, "population_fixed_by_caller": True}
    joined = joined.loc[included].copy()
    joined["observed_channels"] = joined.events_count.gt(0).astype("int16")
    joined["alarm_channels"] = joined.alarm_count.gt(0).astype("int16")
    joined["fault_channels"] = joined.fault_count.gt(0).astype("int16")
    joined["no_power_channels"] = joined.no_power_count.gt(0).astype("int16")
    sums = COUNTS + ["observed_channels", "alarm_channels", "fault_channels", "no_power_channels"]
    daily = joined.groupby(["object_id", "date"], observed=True)[sums].sum().reset_index()
    dates = pd.date_range(start_date, end_date)
    grid = pd.MultiIndex.from_product([identities, dates], names=["object_id", "date"]).to_frame(index=False)
    mart = grid.merge(daily, on=["object_id", "date"], how="left", validate="one_to_one")
    mart[sums] = mart[sums].fillna(0).astype("float32")
    counts = eligible_meta.groupby("object_id").channel_id.nunique()
    mart["catalog_channels"] = mart.object_id.map(counts).astype("float32")
    mart = mart.merge(objects[["object_id", "object_kind", "parent_id"]], on="object_id", how="left", validate="many_to_one")
    mart = mart.sort_values(["object_id", "date"]).reset_index(drop=True)
    mart["alarm_today"] = mart.alarm_channels.gt(0).astype("float32")
    mart["observed_today"] = mart.observed_channels.gt(0).astype("float32")
    mart["observed_fraction"] = mart.observed_channels / mart.catalog_channels
    mart["alarm_channel_fraction"] = mart.alarm_channels / mart.observed_channels.replace(0, np.nan)
    for column in ["object_id", "object_kind", "parent_id"]:
        mart[column] = mart[column].fillna("unknown").astype("category")
    groups = mart.groupby("object_id", observed=True, sort=False)
    for column in ["alarm_today", "alarm_channels", "observed_channels", "fault_channels", "no_power_channels", "events_count"]:
        for days in [1, 2, 7, 14]:
            mart[f"{column}_lag{days}"] = groups[column].shift(days).astype("float32")
        for window in [3, 7, 28]:
            mart[f"{column}_mean{window}"] = groups[column].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True).astype("float32")
    ordinal = groups.cumcount().to_numpy()
    mart["historical_alarm_frequency"] = groups.alarm_today.cumsum() / (ordinal + 1)
    a = mart.alarm_today.to_numpy().reshape(len(identities), len(dates)).astype(bool)
    position = np.arange(len(dates))[None, :]
    last_clear = np.maximum.accumulate(np.where(~a, position, -1), axis=1)
    last_alarm = np.maximum.accumulate(np.where(a, position, -1), axis=1)
    mart["alarm_streak"] = np.where(a, position-last_clear, 0).ravel().astype("float32")
    mart["days_since_alarm"] = np.where(last_alarm < 0, -1, position-last_alarm).ravel().astype("float32")
    mart["target_dayofweek"] = (mart.date + pd.Timedelta(days=2)).dt.dayofweek.astype("int8")
    mart["annual_sin"] = np.sin(2*np.pi*mart.date.dt.dayofyear/365.25).astype("float32")
    mart["annual_cos"] = np.cos(2*np.pi*mart.date.dt.dayofyear/365.25).astype("float32")
    groups = mart.groupby("object_id", observed=True, sort=False)
    future_date = groups.date.shift(-2)
    available = future_date.notna()
    assert future_date[available].eq(mart.loc[available, "date"]+pd.Timedelta(days=2)).all()
    mart[TARGET] = groups.alarm_today.shift(-2)
    coverage["objects_with_observations_in_window"] = int(mart.loc[mart.observed_today.eq(1), "object_id"].nunique())
    return mart, future_date, coverage
