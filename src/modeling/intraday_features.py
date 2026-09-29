#!/usr/bin/env python3
"""Causal, end-of-day timing features from streamed event journals.

All features describe records on the feature day only. They contain no labels
and may be joined to the existing day/channel matrix without changing its target.
First/last records are ordered by second of day, numeric event id, then raw row
position; timestamp collisions are retained and counted, not silently deduplicated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .daily_data import ALARM_TRUE, ROOT, open_archive_csv
except ImportError:
    from daily_data import ALARM_TRUE, ROOT, open_archive_csv

KEY = ["channel_id", "date"]
RAW_COLUMNS = ["ид_события", "ид_канала_данных", "дата", "время", "тревожное"]
SUMS = ["intra_alarm_count_last6h", "intra_alarm_count_last12h", "intra_events_last6h"]


def summarize_chunk(raw: pd.DataFrame, row_offset: int = 0):
    """Return compact partial aggregates and parsing counters for a single chunk."""
    frame = pd.DataFrame({
        "channel_id": raw["ид_канала_данных"],
        "date": pd.to_datetime(raw["дата"], errors="coerce").dt.normalize(),
        "second": pd.to_timedelta(raw["время"], errors="coerce").dt.total_seconds(),
        "event_id": pd.to_numeric(raw["ид_события"], errors="coerce"),
        "_row": np.arange(row_offset, row_offset + len(raw), dtype=np.int64),
    })
    alarm_text = raw["тревожное"].fillna("").str.strip().str.lower()
    frame["alarm"] = alarm_text.isin(ALARM_TRUE).astype("int8")
    valid = (frame.channel_id.notna() & frame.date.notna()
             & frame.second.ge(0) & frame.second.lt(86400))
    stats = {"input_rows": len(raw), "invalid_datetime_or_channel": int((~valid).sum()),
             "invalid_event_id": int(frame.event_id.isna().sum()),
             "unknown_alarm_text": int((~alarm_text.isin(ALARM_TRUE | {"false", "f", "0", "no", "n", "нет", "ложь"})).sum())}
    frame = frame.loc[valid].copy()
    if frame.empty:
        return pd.DataFrame(), stats
    # Missing ids sort before numeric ids; row order remains a deterministic final tie-break.
    frame["event_id"] = frame.event_id.fillna(-1).astype("int64")
    frame["intra_alarm_count_last6h"] = (frame.alarm.eq(1) & frame.second.ge(18 * 3600)).astype("int32")
    frame["intra_alarm_count_last12h"] = (frame.alarm.eq(1) & frame.second.ge(12 * 3600)).astype("int32")
    frame["intra_events_last6h"] = frame.second.ge(18 * 3600).astype("int32")
    frame["intra_last_alarm_second"] = frame.second.where(frame.alarm.eq(1), -1)
    grouped = frame.groupby(KEY, observed=True, sort=False)
    last_second = grouped.second.transform("max")
    frame["_last_timestamp_events"] = frame.second.eq(last_second).astype("int32")
    # Recreate the GroupBy after adding the timestamp-counter column.
    aggregate = frame.groupby(KEY, observed=True, sort=False).agg(
        **{c: (c, "sum") for c in SUMS},
        intra_last_alarm_second=("intra_last_alarm_second", "max"),
        _last_timestamp_events=("_last_timestamp_events", "sum"),
    )
    ordered = frame.sort_values(["second", "event_id", "_row"], kind="stable")
    for name, keep in [("first", "first"), ("last", "last")]:
        selected = ordered.drop_duplicates(KEY, keep=keep).set_index(KEY)
        for column in ["second", "event_id", "_row", "alarm"]:
            aggregate[f"_{name}_{column}"] = selected[column]
    return aggregate.reset_index(), stats


def combine_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Associative reduction: identical results regardless of chunk boundaries."""
    parts = [p for p in parts if not p.empty]
    if not parts:
        raise ValueError("No valid event records")
    partial = pd.concat(parts, ignore_index=True)
    grouped = partial.groupby(KEY, observed=True, sort=False)
    aggregate = grouped.agg(**{c: (c, "sum") for c in SUMS},
                            intra_last_alarm_second=("intra_last_alarm_second", "max"))
    global_last = grouped._last_second.transform("max")
    partial["_global_last_timestamp_events"] = partial._last_timestamp_events.where(
        partial._last_second.eq(global_last), 0)
    aggregate["intra_last_timestamp_ties"] = (
        partial.groupby(KEY, observed=True, sort=False)._global_last_timestamp_events.sum() - 1)
    for name, keep in [("first", "first"), ("last", "last")]:
        selected = partial.sort_values(
            [f"_{name}_second", f"_{name}_event_id", f"_{name}__row"], kind="stable"
        ).drop_duplicates(KEY, keep=keep).set_index(KEY)
        aggregate[f"intra_{name}_event_minute"] = selected[f"_{name}_second"] / 60.0
        aggregate[f"intra_{name}_event_is_alarm"] = selected[f"_{name}_alarm"].astype("int8")
    aggregate["intra_last_alarm_minute"] = (aggregate.pop("intra_last_alarm_second") / 60.0)
    # -1 marks no alarm on the day; do not confuse midnight with no alarm.
    aggregate.loc[aggregate.intra_last_alarm_minute < 0, "intra_last_alarm_minute"] = -1
    for column in aggregate:
        if "minute" in column:
            aggregate[column] = aggregate[column].astype("float32")
        elif "is_alarm" not in column:
            aggregate[column] = aggregate[column].astype("int32")
    result = aggregate.reset_index().sort_values(KEY).reset_index(drop=True)
    if result.duplicated(KEY).any():
        raise AssertionError("Duplicate day/channel rows after aggregation")
    return result


def build_year(year: str, chunksize: int = 500_000, force: bool = False) -> Path:
    archive = ROOT / "data/raw" / f"ext-journal-{year}.7z"
    output = ROOT / "data/interim" / f"intraday_{year}.parquet"
    metadata = output.with_suffix(".json")
    source_stat = archive.stat()
    fingerprint = hashlib.sha256(Path(__file__).read_bytes() + json.dumps({
        "archive": archive.name, "bytes": source_stat.st_size,
        "mtime_ns": source_stat.st_mtime_ns,
    }, sort_keys=True).encode()).hexdigest()
    if not force and output.exists() and metadata.exists():
        if json.loads(metadata.read_text()).get("fingerprint") == fingerprint:
            print(f"[intraday] {year}: valid cache {output.name}", flush=True)
            return output
    start = time.monotonic()
    totals: dict[str, int] = {}
    parts = []
    row_offset = 0
    proc = open_archive_csv(archive)
    try:
        assert proc.stdout is not None
        for number, raw in enumerate(pd.read_csv(proc.stdout, dtype=str, usecols=RAW_COLUMNS,
                                                chunksize=chunksize), start=1):
            part, stats = summarize_chunk(raw, row_offset)
            parts.append(part)
            row_offset += len(raw)
            for key, value in stats.items():
                totals[key] = totals.get(key, 0) + value
            if number % 5 == 0:
                print(f"[intraday] {year}: {row_offset:,} rows, {time.monotonic()-start:.1f}s", flush=True)
        proc.wait()
        if proc.returncode:
            error = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
            raise RuntimeError(f"Cannot read {archive.name}: {error}")
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    result = combine_parts(parts)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.parquet")
    result.to_parquet(temporary, index=False)
    temporary.replace(output)
    totals.update({"channel_days": len(result),
                   "days_with_last_timestamp_collisions": int(result.intra_last_timestamp_ties.gt(0).sum()),
                   "extra_records_at_last_timestamp": int(result.intra_last_timestamp_ties.sum()),
                   "max_last_timestamp_ties": int(result.intra_last_timestamp_ties.max())})
    metadata.write_text(json.dumps({
        "year": year, "fingerprint": fingerprint, "source_archive": archive.name,
        "source_bytes": source_stat.st_size, "source_mtime_ns": source_stat.st_mtime_ns,
        "columns": result.columns.tolist(), "diagnostics": totals,
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "tie_policy": "second of day, numeric event id, original raw row position",
        "cutoff": "Feature day end; contains no next-day or future observations",
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"[intraday] {year}: saved {len(result):,} rows in {time.monotonic()-start:.1f}s", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", default=["2025", "2026"])
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for year in args.years:
        build_year(year, args.chunksize, args.force)
