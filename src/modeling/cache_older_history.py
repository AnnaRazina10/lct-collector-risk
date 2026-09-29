#!/usr/bin/env python3
"""Stream 2019–2024 journals into sparse daily caches and pre-2025 priors.

Run: .venv/bin/python -u src/modeling/cache_older_history.py
No model is trained. Only observed channel/day rows are stored. Priors are
incrementally rebuilt from successfully verified years, always before 2025.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

try:
    from . import daily_data
except ImportError:
    import daily_data

ROOT = daily_data.ROOT
CACHE = ROOT / "data/interim"
OUT = ROOT / "reports/improved_v2"
SCHEMA_VERSION = 1
CUTOFF = pd.Timestamp("2025-01-01")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def log(message):
    print(time.strftime("%H:%M:%S"), "[history]", message, flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def write_parquet(path, frame):
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def validate_daily(daily, year, schema):
    if list(daily.columns) != list(schema):
        raise ValueError(f"Year {year}: daily schema differs from daily_2025.parquet")
    if daily.empty or daily[["channel_id", "date"]].isna().any().any():
        raise ValueError(f"Year {year}: empty data or missing keys")
    if daily.duplicated(["channel_id", "date"]).any():
        raise ValueError(f"Year {year}: duplicate channel/day rows")
    if not daily.date.eq(daily.date.dt.normalize()).all():
        raise ValueError(f"Year {year}: non-daily timestamps")
    if not daily.date.dt.year.eq(year).all() or not daily.date.lt(CUTOFF).all():
        raise ValueError(f"Year {year}: timestamp outside requested historical year")
    for column, dtype in schema.items():
        if column.endswith("_count"):
            if daily[column].isna().any() or daily[column].lt(0).any():
                raise ValueError(f"Year {year}: negative/missing {column}")
            # Preserve the template where possible without narrowing large counts.
            if pd.api.types.is_integer_dtype(dtype):
                limit = np.iinfo(dtype)
                target_dtype = dtype if daily[column].max() <= limit.max else "int64"
                daily[column] = daily[column].astype(target_dtype)
        else:
            daily[column] = daily[column].astype(dtype)
    for column in ["alarm_count", "numeric_count", "fault_count", "no_power_count"]:
        if daily[column].gt(daily.events_count).any():
            raise ValueError(f"Year {year}: {column} exceeds events_count")
    parts = daily[["numeric_count", "text_count", "missing_count"]].astype("int64").sum(axis=1)
    if not parts.eq(daily.events_count).all():
        raise ValueError(f"Year {year}: numeric/text/missing counts do not reconcile with events_count")
    return daily.sort_values(["channel_id", "date"], ignore_index=True)


def year_profile(daily, year):
    counts = [column for column in daily if column.endswith("_count")]
    # int64 accumulation avoids overflow across many dates/years.
    values = daily[["channel_id", "date"] + counts + ["value_sum", "value_sumsq"]].copy()
    values[counts] = values[counts].astype("int64")
    values["alarm_days"] = daily.alarm_count.gt(0).astype("int64")
    values["fault_days"] = daily.fault_count.gt(0).astype("int64")
    values["no_power_days"] = daily.no_power_count.gt(0).astype("int64")
    values["last_alarm"] = daily.date.where(daily.alarm_count.gt(0))
    aggregation = {column: "sum" for column in counts +
                   ["value_sum", "value_sumsq", "alarm_days", "fault_days", "no_power_days"]}
    result = values.groupby("channel_id", sort=True).agg(
        **{column: (column, method) for column, method in aggregation.items()},
        observed_days=("date", "size"),
        first_observed=("date", "min"), last_observed=("date", "max"),
        last_alarm=("last_alarm", "max"),
    ).reset_index()
    result["year"] = year
    return result


def build_prior(profiles):
    combined = pd.concat(profiles, ignore_index=True)
    sums = [column for column in combined if column not in
            {"channel_id", "year", "first_observed", "last_observed", "last_alarm"}]
    aggregation = {column: (column, "sum") for column in sums}
    prior = combined.groupby("channel_id", sort=True).agg(
        **aggregation,
        first_observed=("first_observed", "min"),
        last_observed=("last_observed", "max"),
        last_alarm=("last_alarm", "max"),
        observed_years=("year", "nunique"),
    ).reset_index()
    prior["alarm_event_rate"] = prior.alarm_count / prior.events_count
    prior["alarm_observed_day_rate"] = prior.alarm_days / prior.observed_days
    prior["fault_observed_day_rate"] = prior.fault_days / prior.observed_days
    prior["no_power_observed_day_rate"] = prior.no_power_days / prior.observed_days
    prior["events_per_observed_day"] = prior.events_count / prior.observed_days
    prior["days_since_last_observed_at_cutoff"] = (CUTOFF - prior.last_observed).dt.days
    numeric_n = prior.numeric_count.replace(0, np.nan)
    prior["numeric_mean"] = prior.value_sum / numeric_n
    prior["numeric_std"] = np.sqrt((prior.value_sumsq / numeric_n - prior.numeric_mean**2).clip(lower=0))
    model_prior = pd.DataFrame({
        "channel_id": prior.channel_id.astype("str"),
        "old_observed_days": prior.observed_days,
        "old_alarm_days": prior.alarm_days,
        "old_fault_days": prior.fault_days,
        "old_no_power_days": prior.no_power_days,
        "old_events": prior.events_count,
        "old_alarm_events": prior.alarm_count,
        "old_years_observed": prior.observed_years,
        "old_last_observed_days_before_2025": (CUTOFF - prior.last_observed).dt.days,
        "old_last_alarm_days_before_2025": (CUTOFF - prior.last_alarm).dt.days.fillna(-1).astype("int64"),
        "old_alarm_days_per_observed_day": (prior.alarm_days + 1) / (prior.observed_days + 2),
    })
    years = sorted(int(profile.year.iloc[0]) for profile in profiles)
    write_parquet(CACHE / "channel_prior_pre2025.parquet", model_prior)
    if years == [2024]:
        write_parquet(CACHE / "channel_prior_2024.parquet", model_prior)
    write_json(CACHE / "channel_prior_pre2025.metadata.json", {
        "years": years, "cutoff_exclusive": str(CUTOFF.date()),
        "channels": len(model_prior), "columns": list(model_prior),
        "smoothing": "Beta(1,1): (alarm_days + 1) / (observed_days + 2)",
        "missing_last_alarm": -1,
        "scope": "Observed historical messages before 2025 only; no target2025/2026",
    })
    prior = prior.rename(columns={column: "history_" + column for column in prior if column != "channel_id"})
    write_parquet(CACHE / "channel_history_prior_pre2025.parquet", prior)
    return len(prior)


def save_summary(records, profiles, requested_years):
    rows = sorted(records.values(), key=lambda row: row["year"])
    prior_channels = build_prior(profiles) if profiles else 0
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "requested_years": requested_years,
        "completed_years": sorted(int(profile.year.iloc[0]) for profile in profiles),
        "prior_cutoff_exclusive": str(CUTOFF.date()),
        "prior_channels": prior_channels,
        "prior_path": "data/interim/channel_prior_pre2025.parquet" if profiles else None,
        "record_count_definition": "Sum of events_count after dropping invalid date/channel rows; not raw CSV line count",
        "coverage_note": "Sparse observed channel/day rows only; no complete multi-year grid. Prior contains only completed_years.",
        "target_note": "Alarm and fault-message history is not a label of confirmed equipment failure.",
        "years": rows,
    }
    write_json(OUT / "history_summary.json", payload)
    pd.DataFrame(rows).to_csv(OUT / "history_summary.csv", index=False)


def run(args):
    manifest_path = daily_data.RAW / "import_manifest.json"
    manifest = {item["name"]: item for item in json.loads(manifest_path.read_text())}
    template = pd.read_parquet(CACHE / "daily_2025.parquet")
    schema = template.dtypes.to_dict()
    del template
    records, profiles = {}, []
    aggregator_sha = sha256(Path(daily_data.__file__))
    for year in args.years:
        start = time.monotonic()
        archive = daily_data.RAW / f"ext-journal-{year}.7z"
        log(f"Start {year}; validating archive against import manifest")
        try:
            expected = manifest.get(archive.name)
            if expected is None or not archive.is_file():
                raise ValueError(f"Archive or manifest record missing: {archive.name}")
            if archive.stat().st_size != expected["bytes"]:
                raise ValueError(f"Archive size mismatch: {archive.name}")
            actual_sha = sha256(archive)
            if actual_sha != expected["sha256"]:
                raise ValueError(f"Archive SHA256 mismatch: {archive.name}")
            stamp = {"schema_version": SCHEMA_VERSION, "archive_sha256": actual_sha,
                     "aggregator_sha256": aggregator_sha,
                     "schema": {key: str(dtype) for key, dtype in schema.items()}}
            path = CACHE / f"daily_{year}.parquet"
            metadata_path = CACHE / f"daily_{year}.history_manifest.json"
            cached = path.exists() and metadata_path.exists() and json.loads(metadata_path.read_text()) == stamp
            daily = pd.read_parquet(path) if cached else daily_data.read_year_daily(str(year), args.chunksize)
            daily = validate_daily(daily, year, schema)
            if not cached:
                write_parquet(path, daily)
                write_json(metadata_path, stamp)
            profile = year_profile(daily, year)
            write_parquet(CACHE / f"channel_history_{year}.parquet", profile)
            profiles.append(profile)
            records[year] = {
                "year": year, "status": "cached" if cached else "aggregated",
                "daily_rows": len(daily), "channels": int(daily.channel_id.nunique()),
                "valid_records": int(daily.events_count.astype("int64").sum()),
                "alarm_records": int(daily.alarm_count.astype("int64").sum()),
                "fault_records": int(daily.fault_count.astype("int64").sum()),
                "first_date": str(daily.date.min().date()), "last_date": str(daily.date.max().date()),
                "archive_bytes": archive.stat().st_size, "archive_sha256": actual_sha,
                "seconds": round(time.monotonic() - start, 2),
            }
            del daily
            gc.collect()
            save_summary(records, profiles, args.years)
            log(f"READY {year}: {records[year]['daily_rows']:,} channel/day rows, "
                f"{records[year]['channels']:,} channels, {records[year]['valid_records']:,} valid records; "
                f"{records[year]['seconds']:.1f}s. Prior updated.")
        except Exception as error:
            records[year] = {"year": year, "status": "error", "error": str(error),
                             "seconds": round(time.monotonic() - start, 2)}
            save_summary(records, profiles, args.years)
            log(f"ERROR {year}: {error}")
            if args.fail_fast:
                raise
    return 1 if any(row["status"] == "error" for row in records.values()) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2024, 2018, -1)))
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if any(year < 2019 or year > 2024 for year in args.years) or len(set(args.years)) != len(args.years):
        parser.error("Years must be distinct and within 2019–2024")
    if args.chunksize < 1:
        parser.error("chunksize must be positive")
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "history_progress.log").open("a", buffering=1) as handle:
        with redirect_stdout(Tee(sys.stdout, handle)), redirect_stderr(Tee(sys.stderr, handle)):
            return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
