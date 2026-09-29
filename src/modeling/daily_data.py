"""Streaming raw journal aggregation, independent of ML dependencies."""
from __future__ import annotations
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw"

ALARM_TRUE = {"true", "t", "1", "yes", "y", "да", "истина"}
SENTINELS = {-100.0, 100.0, 255.0, 999.0, 9999.0, -999.0}
TEXT_FLAGS = {
    "fault_count": "неисправен",
    "no_power_count": "обесточен",
    "undefined_count": "неопределен",
    "off_count": "выключен",
    "on_count": "включен",
    "movement_detected_count": "обнаружено движение",
    "no_movement_count": "движения нет",
    "norm_count": "норма",
    "open_count": "не замкнут",
}


def open_archive_csv(archive_path: Path) -> subprocess.Popen[bytes]:
    inner = archive_path.with_suffix(".csv").name
    proc = subprocess.Popen(
        ["bsdtar", "-xOf", str(archive_path), inner],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.stdout is None:
        raise RuntimeError(f"Cannot read {archive_path}")
    return proc


def load_dictionaries() -> pd.DataFrame:
    channels = pd.read_csv(RAW / "справочник_каналов_датчиков.csv", dtype=str)
    objects = pd.read_csv(RAW / "справочник_объектов_диспетчер.csv", dtype=str)
    meta = channels.merge(objects, on="ид_объект", how="left")
    meta = meta.rename(
        columns={
            "ид_канала_данных": "channel_id",
            "тип_инж_системы": "engineering_system",
            "тип_датчика": "sensor_type",
            "ид_объект": "object_id",
            "вид_объекта": "object_kind",
        }
    )
    keep = ["channel_id", "engineering_system", "sensor_type", "object_id", "object_kind"]
    return meta[keep].drop_duplicates("channel_id")


def combine_daily(parts: list[pd.DataFrame]) -> pd.DataFrame:
    daily = pd.concat(parts, ignore_index=True)
    sum_cols = [col for col in daily.columns if col not in {"channel_id", "date", "value_min", "value_max"}]
    agg = {col: "sum" for col in sum_cols}
    agg["value_min"] = "min"
    agg["value_max"] = "max"
    return daily.groupby(["channel_id", "date"], observed=True).agg(agg).reset_index()


def read_year_daily(year: str, chunksize: int) -> pd.DataFrame:
    archive = RAW / f"ext-journal-{year}.7z"
    if not archive.exists():
        raise FileNotFoundError(f"Missing raw archive: {archive}")

    print(f"[baseline] aggregate {archive.name}")
    proc = open_archive_csv(archive)
    assert proc.stdout is not None
    chunks: list[pd.DataFrame] = []

    for chunk_number, chunk in enumerate(pd.read_csv(proc.stdout, chunksize=chunksize, dtype=str), start=1):
        chunk = chunk.rename(
            columns={
                "ид_события": "event_id",
                "ид_канала_данных": "channel_id",
                "дата": "date",
                "время": "time",
                "тревожное": "alarm_raw",
                "значение_датчика": "sensor_value",
            }
        )
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        chunk = chunk.dropna(subset=["date", "channel_id"])
        chunk["channel_id"] = chunk["channel_id"].astype(str)
        alarm_raw = chunk["alarm_raw"].fillna("").str.strip().str.lower()
        chunk["is_alarm"] = alarm_raw.isin(ALARM_TRUE).astype("int16")

        value_text = chunk["sensor_value"].fillna("").astype(str).str.strip()
        value_num = pd.to_numeric(value_text.str.replace(",", ".", regex=False), errors="coerce")
        is_missing = value_text.eq("")
        is_numeric = value_num.notna() & ~is_missing
        is_text = ~is_numeric & ~is_missing

        chunk["event_count"] = np.int16(1)
        chunk["numeric_count"] = is_numeric.astype("int16")
        chunk["text_count"] = is_text.astype("int16")
        chunk["missing_count"] = is_missing.astype("int16")
        chunk["negative_count"] = (value_num < 0).fillna(False).astype("int16")
        chunk["sentinel_count"] = value_num.isin(SENTINELS).astype("int16")
        chunk["numeric_value"] = value_num.astype("float32")
        chunk["value_sum"] = value_num.fillna(0).astype("float64")
        chunk["value_sumsq"] = value_num.fillna(0).pow(2).astype("float64")

        lowered = value_text.str.lower()
        for col, text in TEXT_FLAGS.items():
            chunk[col] = lowered.eq(text).astype("int16")

        grouped = (
            chunk.groupby(["channel_id", "date"], observed=True)
            .agg(
                events_count=("event_count", "sum"),
                alarm_count=("is_alarm", "sum"),
                numeric_count=("numeric_count", "sum"),
                text_count=("text_count", "sum"),
                missing_count=("missing_count", "sum"),
                negative_count=("negative_count", "sum"),
                sentinel_count=("sentinel_count", "sum"),
                value_sum=("value_sum", "sum"),
                value_sumsq=("value_sumsq", "sum"),
                value_min=("numeric_value", "min"),
                value_max=("numeric_value", "max"),
                **{col: (col, "sum") for col in TEXT_FLAGS},
            )
            .reset_index()
        )
        chunks.append(grouped)
        if chunk_number % 5 == 0:
            print(f"[baseline] {year}: {chunk_number * chunksize:,} rows read", flush=True)

    proc.wait()
    if proc.returncode != 0:
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        raise RuntimeError(f"bsdtar failed for {archive.name}: {stderr}")

    return combine_daily(chunks)
