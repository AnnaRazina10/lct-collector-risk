"""Fetch public Moscow ERA5 context for a separate research-only ablation.

No project data are read or sent. A seven-calendar-day lag is a feature policy,
not evidence that the current reanalysis vintage existed at historical time D.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
VARIABLES = (
    "temperature_2m_mean", "temperature_2m_min", "temperature_2m_max",
    "precipitation_sum", "relative_humidity_2m_mean", "pressure_msl_mean",
    "surface_pressure_mean",
)
STEM = "московскийконтекст"
LAG_DAYS = 7
DOCS = "https://open-meteo.com/en/docs/historical-weather-api"
ERA5 = "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def csv_bytes(rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def validate_and_rows(payload: dict, start: date, end: date) -> list[dict]:
    if payload.get("error"):
        raise ValueError(f"Weather API error: {payload.get('reason')}")
    daily = payload["daily"]
    expected = [(start + timedelta(days=i)).isoformat()
                for i in range((end - start).days + 1)]
    if daily["time"] != expected:
        raise ValueError("API date coverage is not complete, ordered and unique")
    if payload["timezone"] != "Europe/Moscow" or payload["utc_offset_seconds"] != 10800:
        raise ValueError("Unexpected daily aggregation timezone")
    for name in VARIABLES:
        values = daily[name]
        if len(values) != len(expected):
            raise ValueError(f"Length mismatch for {name}")
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError(f"Missing/non-finite weather values in {name}")
    expected_units = {name: "°C" for name in VARIABLES if name.startswith("temperature")}
    expected_units.update(precipitation_sum="mm", relative_humidity_2m_mean="%",
                          pressure_msl_mean="hPa", surface_pressure_mean="hPa")
    if any(payload["daily_units"].get(k) != v for k, v in expected_units.items()):
        raise ValueError("Unexpected units")
    rows = [{"source_date": day, **{name: daily[name][i] for name in VARIABLES}}
            for i, day in enumerate(expected)]
    for row in rows:
        if not row["temperature_2m_min"] <= row["temperature_2m_mean"] <= row["temperature_2m_max"]:
            raise ValueError("Temperature ordering failed")
        if row["precipitation_sum"] < 0 or not 0 <= row["relative_humidity_2m_mean"] <= 100:
            raise ValueError("Invalid precipitation or humidity")
        if row["pressure_msl_mean"] <= 0 or row["surface_pressure_mean"] <= 0:
            raise ValueError("Invalid pressure")
    return rows


def delayed_rows(rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        source = date.fromisoformat(row["source_date"])
        feature_day = source + timedelta(days=LAG_DAYS)
        result.append({"date": feature_day.isoformat(), "source_date": source.isoformat(),
                       **{f"weather_lag7_{v}": row[v] for v in VARIABLES}})
    # Check the join contract by keys, not positional alignment or row offsets.
    source_by_date = {r["source_date"]: r for r in rows}
    if len({r["date"] for r in result}) != len(result):
        raise ValueError("Duplicate feature dates")
    for row in result:
        feature_day = date.fromisoformat(row["date"])
        source = date.fromisoformat(row["source_date"])
        if source != feature_day - timedelta(days=LAG_DAYS):
            raise ValueError("Weather lag contract failed")
        if any(row[f"weather_lag7_{v}"] != source_by_date[source.isoformat()][v] for v in VARIABLES):
            raise ValueError("Weather source-value alignment failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2024-12-01", type=date.fromisoformat)
    parser.add_argument("--end", default="2026-06-30", type=date.fromisoformat)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/raw/external")
    parser.add_argument("--refresh", action="store_true", help="Explicitly replace the saved API vintage")
    args = parser.parse_args()
    if args.end < args.start:
        parser.error("end must be on or after start")
    params = {"latitude": "55.75", "longitude": "37.62",
              "start_date": args.start.isoformat(), "end_date": args.end.isoformat(),
              "daily": ",".join(VARIABLES), "models": "era5", "timezone": "Europe/Moscow",
              "temperature_unit": "celsius", "precipitation_unit": "mm"}
    url = "https://archive-api.open-meteo.com/v1/archive?" + urlencode(params)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / f"{STEM}.json"
    provenance_path = out / f"{STEM}.provenance.json"
    cached = raw_path.exists() and not args.refresh
    if cached:
        previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        raw = raw_path.read_bytes()
        if previous["api_url"] != url or previous["files"][raw_path.name]["sha256"] != digest(raw):
            raise ValueError("Cached request or SHA256 differs; inspect before an explicit --refresh")
        retrieved_at = previous["retrieved_at_utc"]
        response_headers = previous.get("response_headers", {})
    else:
        print(f"Fetching public Moscow context: {args.start}..{args.end}, ERA5", flush=True)
        request = Request(url, headers={"User-Agent": "lct-collector-risk-weather-research/1.0"})
        with urlopen(request, timeout=90) as response:
            raw = response.read()
            response_headers = {k: response.headers.get(k) for k in ("Date", "Last-Modified", "ETag")}
        retrieved_at = datetime.now(timezone.utc).isoformat()
    payload = json.loads(raw)
    rows = validate_and_rows(payload, args.start, args.end)
    features = delayed_rows(rows)
    outputs = {raw_path.name: raw, f"{STEM}.csv": csv_bytes(rows),
               f"{STEM}_lag7.csv": csv_bytes(features)}
    for filename, data in outputs.items():
        (out / filename).write_bytes(data)
    provenance = {
        "api_url": url, "request_parameters": params, "model_requested": "era5",
        "model_response_identification": "Single model fixed by request; API does not expose vintage ID",
        "retrieved_at_utc": retrieved_at, "response_headers": response_headers,
        "requested_location": {"latitude": 55.75, "longitude": 37.62, "scope": "generic Moscow context; not object coordinates"},
        "returned_grid": {k: payload.get(k) for k in ("latitude", "longitude", "elevation", "timezone", "utc_offset_seconds")},
        "source_dates": {"first": rows[0]["source_date"], "last": rows[-1]["source_date"], "rows": len(rows)},
        "feature_dates": {"first": features[0]["date"], "last": features[-1]["date"], "lag_calendar_days": LAG_DAYS},
        "daily_units": payload["daily_units"], "api_documentation": DOCS,
        "license": {"name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/",
                    "source": "https://open-meteo.com/en/licence", "data_attribution": "Weather data by Open-Meteo.com (https://open-meteo.com/), based on ERA5 / Copernicus Climate Change Service"},
        "citations": [{"title": "Zippenfenig, P. (2023). Open-Meteo.com Weather API [Computer software]. Zenodo.", "url": "https://doi.org/10.5281/zenodo.7970649"},
                      {"title": "ERA5 hourly data on single levels from 1940 to present", "url": "https://doi.org/10.24381/cds.adbb2d47", "documentation": ERA5}],
        "changes": "JSON retained byte-for-byte; daily CSV extracted; feature dates shifted forward by 7 calendar days and weather columns renamed",
        "suitability": {"research_ablation_only": True, "strict_point_in_time": False,
                        "historical_available_at_known": False, "revision_vintage_known": False,
                        "reason": "Current reanalysis vintage may differ from historical ERA5T; a seven-day lag does not reconstruct the past published vintage"},
        "validation": {"continuous_dates": True, "finite_values": True, "physical_bounds": True,
                       "unique_feature_dates": True, "keyed_lag7_source_alignment": True},
        "files": {name: {"sha256": digest(data), "bytes": len(data)} for name, data in outputs.items()},
    }
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "cached": cached, "feature_first": features[0]["date"],
                      "feature_last": features[-1]["date"], "output_dir": str(out)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
