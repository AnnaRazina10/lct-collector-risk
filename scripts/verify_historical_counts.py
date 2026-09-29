#!/usr/bin/env python3
"""Read-only streaming verification of the 11 object-model journal counters.

Recomputes counters independently of daily_data.read_year_daily. It never fits a
model, computes predictive metrics, changes a cache, or re-labels events with the
current state dictionary. Requires the original archives and pandas/pyarrow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COUNTS = ["events_count", "alarm_count", "fault_count", "no_power_count", "undefined_count",
          "numeric_count", "text_count", "off_count", "on_count", "movement_detected_count", "open_count"]
TEXT_FLAGS = {"fault_count": "неисправен", "no_power_count": "обесточен",
              "undefined_count": "неопределен", "off_count": "выключен", "on_count": "включен",
              "movement_detected_count": "обнаружено движение", "open_count": "не замкнут"}
ALARM_TRUE = {"true", "t", "1", "yes", "y", "да", "истина"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(year):
    started = time.monotonic()
    archive = ROOT / f"data/raw/ext-journal-{year}.7z"
    cache = ROOT / f"data/interim/daily_{year}.parquet"
    manifest_path = ROOT / f"data/interim/daily_{year}.history_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    initial_cache_sha, initial_archive_sha = sha256(cache), sha256(archive)
    process = subprocess.Popen(["bsdtar", "-xOf", str(archive), f"ext-journal-{year}.csv"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(json.dumps({"stage": "started", "year": year, "python_pid": os.getpid(),
                      "archive_reader_pid": process.pid, "cache_sha256": initial_cache_sha}), flush=True)
    parts, raw_rows, valid_rows, last_progress = [], 0, 0, 0.0
    try:
        reader = pd.read_csv(process.stdout, dtype=str, chunksize=500_000,
                             usecols=["ид_канала_данных", "дата", "тревожное", "значение_датчика"])
        for number, chunk in enumerate(reader, 1):
            raw_rows += len(chunk)
            dates = pd.to_datetime(chunk["дата"], errors="coerce")
            keep = dates.notna() & chunk["ид_канала_данных"].notna()
            chunk, dates = chunk.loc[keep], dates.loc[keep]
            valid_rows += len(chunk)
            value = chunk["значение_датчика"].fillna("").astype(str).str.strip()
            numeric = pd.to_numeric(value.str.replace(",", ".", regex=False), errors="coerce")
            missing = value.eq("")
            is_numeric = numeric.notna() & ~missing
            counts = pd.DataFrame({
                "channel_id": chunk["ид_канала_данных"].astype(str), "date": dates,
                "events_count": np.ones(len(chunk), dtype="int64"),
                "alarm_count": chunk["тревожное"].fillna("").str.strip().str.lower().isin(ALARM_TRUE).astype("int64"),
                "numeric_count": is_numeric.astype("int64"),
                "text_count": (~is_numeric & ~missing).astype("int64"),
            })
            lowered = value.str.lower()
            for column, label in TEXT_FLAGS.items():
                counts[column] = lowered.eq(label).astype("int64")
            parts.append(counts.groupby(["channel_id", "date"], observed=True)[COUNTS].sum())
            elapsed = time.monotonic() - started
            if elapsed - last_progress >= 12:
                print(json.dumps({"stage": "streaming", "year": year, "chunks": number,
                                  "raw_rows": raw_rows, "valid_rows": valid_rows,
                                  "elapsed_seconds": round(elapsed, 2)}), flush=True)
                last_progress = elapsed
        code = process.wait()
        if code:
            raise RuntimeError(f"bsdtar exited {code}: " + process.stderr.read().decode(errors="replace"))
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
    rebuilt = pd.concat(parts).groupby(level=["channel_id", "date"], observed=True)[COUNTS].sum().sort_index()
    cached = pd.read_parquet(cache, columns=["channel_id", "date", *COUNTS])
    duplicates = int(cached.duplicated(["channel_id", "date"]).sum())
    if duplicates:
        raise ValueError(f"Cached channel-day identifiers are not unique: {duplicates} duplicates")
    cached["channel_id"] = cached.channel_id.astype(str)
    cached = cached.set_index(["channel_id", "date"])[COUNTS].astype("int64").sort_index()
    common = rebuilt.index.intersection(cached.index)
    differences = rebuilt.loc[common] - cached.loc[common]
    keys_equal = rebuilt.index.equals(cached.index)
    column_results = {column: {
        "mismatched_channel_days": int(differences[column].ne(0).sum()),
        "maximum_absolute_difference": int(differences[column].abs().max()) if len(common) else None,
        "rebuilt_total": int(rebuilt[column].sum()), "cached_total": int(cached[column].sum()),
    } for column in COUNTS}
    final_cache_sha, final_archive_sha = sha256(cache), sha256(archive)
    passed = (keys_equal and differences.eq(0).all().all() and initial_cache_sha == final_cache_sha
              and initial_archive_sha == final_archive_sha == manifest["archive_sha256"])
    return {
        "year": year, "status": "passed" if passed else "failed",
        "completed_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "verification": "Independent streaming recomputation of all 11 model COUNTS; original parquet never modified",
        "source_script_path": str(Path(__file__).resolve().relative_to(ROOT)),
        "source_script_sha256": sha256(Path(__file__)),
        "archive_sha256": initial_archive_sha, "archive_sha256_after": final_archive_sha,
        "archive_unchanged": initial_archive_sha == final_archive_sha,
        "archive_hash_matches_history_manifest": final_archive_sha == manifest["archive_sha256"],
        "history_manifest_sha256": sha256(manifest_path),
        "cache_sha256_before": initial_cache_sha, "cache_sha256_after": final_cache_sha,
        "cache_unchanged": initial_cache_sha == final_cache_sha,
        "raw_rows": raw_rows, "valid_rows": valid_rows, "invalid_date_or_channel_rows": raw_rows - valid_rows,
        "rebuilt_channel_days": len(rebuilt), "cached_channel_days": len(cached),
        "duplicate_cached_channel_days": duplicates, "keys_exactly_equal": keys_equal,
        "rebuilt_only_keys": len(rebuilt.index.difference(cached.index)),
        "cached_only_keys": len(cached.index.difference(rebuilt.index)), "columns": column_results,
        "elapsed_seconds": time.monotonic() - started,
        "historical_aggregator_sha256": manifest["aggregator_sha256"],
        "historical_aggregator_source_identity_recovered": False,
        "current_aggregator_sha256": sha256(ROOT / "src/modeling/daily_data.py"),
        "uses_current_state_dictionary_for_relabeling": False,
        "models_trained": False, "model_metrics_computed": False,
        "final_note": "The old aggregator source has not been recovered. Exact raw-to-cache count equality validates the 11 input counters without claiming that historical source identity was recovered.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("year", type=int, choices=[2023, 2024])
    parser.add_argument("--report-path", type=Path,
                        help="JSON receipt destination; default reports/object_onset_backtest/rawYEAR_count_check.json")
    args = parser.parse_args()
    destination = args.report_path or ROOT / f"reports/object_onset_backtest/raw{args.year}_count_check.json"
    result = verify(args.year)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                     suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(destination)
    print(json.dumps({"stage": "finished", "year": args.year, "status": result["status"],
                      "report_path": str(destination), "raw_rows": result["raw_rows"],
                      "channel_days": result["cached_channel_days"],
                      "source_script_sha256": result["source_script_sha256"],
                      "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
