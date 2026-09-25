#!/usr/bin/env python3
"""Streaming EDA for the LCT collector dataset.

The raw journal files are large 7z archives, so the script reads them through
bsdtar without extracting them into the repository.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
REPORT = ROOT / "reports" / "eda"
TABLES = REPORT / "tables"
FIGURES = REPORT / "figures"
RAW_STATS = REPORT / "raw_stats"


ALARM_TRUE = {"true", "t", "1", "yes", "y", "да", "истина"}
ALARM_FALSE = {"false", "f", "0", "no", "n", "нет", "ложь"}
SENTINELS = {-100.0, 100.0, 255.0, 999.0, 9999.0, -999.0}
WEEKDAYS_RU = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]


@dataclass
class NumStats:
    count: int = 0
    total: float = 0.0
    sumsq: float = 0.0
    min_value: float | None = None
    max_value: float | None = None

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.sumsq += value * value
        self.min_value = value if self.min_value is None else min(self.min_value, value)
        self.max_value = value if self.max_value is None else max(self.max_value, value)

    @property
    def avg(self) -> float | None:
        return self.total / self.count if self.count else None

    @property
    def std(self) -> float | None:
        if self.count <= 1:
            return None
        variance = max(0.0, self.sumsq / self.count - (self.total / self.count) ** 2)
        return math.sqrt(variance)


@dataclass
class ChannelStats:
    rows: int = 0
    alarms: int = 0
    numeric: int = 0
    text: int = 0
    missing_value: int = 0
    negative: int = 0
    sentinel: int = 0
    first_ord: int | None = None
    last_ord: int | None = None
    value_stats: NumStats = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.value_stats is None:
            self.value_stats = NumStats()


class HyperLogLog:
    """Small HyperLogLog for approximate duplicate checks."""

    def __init__(self, p: int = 14) -> None:
        self.p = p
        self.m = 1 << p
        self.registers = [0] * self.m
        self.mask = self.m - 1

    @staticmethod
    def _splitmix64(x: int) -> int:
        x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = x
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def add_int(self, value: int) -> None:
        x = self._splitmix64(value)
        idx = x & self.mask
        w = x >> self.p
        width = 64 - self.p
        if w == 0:
            rho = width + 1
        else:
            rho = width - w.bit_length() + 1
        if rho > self.registers[idx]:
            self.registers[idx] = rho

    def estimate(self) -> float:
        m = self.m
        alpha = 0.7213 / (1 + 1.079 / m)
        indicator = sum(2.0 ** -r for r in self.registers)
        raw = alpha * m * m / indicator
        zeros = self.registers.count(0)
        if raw <= 2.5 * m and zeros:
            return m * math.log(m / zeros)
        return raw


def ensure_dirs() -> None:
    for path in [REPORT, TABLES, FIGURES, RAW_STATS]:
        path.mkdir(parents=True, exist_ok=True)


def parse_alarm(value: str) -> tuple[bool | None, str]:
    v = value.strip().lower()
    if v in ALARM_TRUE:
        return True, "ok"
    if v in ALARM_FALSE:
        return False, "ok"
    if not v:
        return None, "missing"
    return None, "invalid"


def parse_float(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value.replace(",", "."))
    except ValueError:
        return None


def pct(part: int | float, total: int | float) -> float:
    return 100.0 * part / total if total else 0.0


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key, "")) for key in fieldnames})


def markdown_table(rows: list[dict[str, object]], columns: list[str], limit: int | None = None) -> str:
    rows = rows if limit is None else rows[:limit]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def load_dicts() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    channels_path = RAW / "справочник_каналов_датчиков.csv"
    objects_path = RAW / "справочник_объектов_диспетчер.csv"
    channels: dict[str, dict[str, str]] = {}
    objects: dict[str, dict[str, str]] = {}
    with channels_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            channels[row["ид_канала_данных"]] = row
    with objects_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            objects[row["ид_объект"]] = row
    return channels, objects


def open_archive_csv(archive_path: Path) -> tuple[subprocess.Popen[bytes], csv.reader]:
    inner = archive_path.with_suffix(".csv").name
    proc = subprocess.Popen(
        ["bsdtar", "-xOf", str(archive_path), inner],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None
    text = io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace", newline="")
    return proc, csv.reader(text)


def svg_bar_chart(
    path: Path,
    labels: list[str],
    values: list[float],
    title: str,
    y_label: str,
    color: str = "#2563eb",
    width: int = 1100,
    height: int = 620,
) -> None:
    margin_l, margin_r, margin_t, margin_b = 90, 30, 70, 140
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    max_v = max(values) if values else 1
    bar_gap = 8
    bar_w = max(4, (plot_w - bar_gap * max(0, len(values) - 1)) / max(1, len(values)))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-family="Arial" font-size="24" font-weight="700">{title}</text>',
        f'<text x="22" y="{margin_t + plot_h/2}" transform="rotate(-90 22 {margin_t + plot_h/2})" text-anchor="middle" font-family="Arial" font-size="15">{y_label}</text>',
        f'<line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" y2="{margin_t + plot_h}" stroke="#333"/>',
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" stroke="#333"/>',
    ]
    for i in range(6):
        y = margin_t + plot_h - plot_h * i / 5
        val = max_v * i / 5
        parts.append(f'<line x1="{margin_l}" y1="{y:.1f}" x2="{margin_l + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{margin_l - 8}" y="{y+5:.1f}" text-anchor="end" font-family="Arial" font-size="12">{val:.2g}</text>')
    for idx, (label, value) in enumerate(zip(labels, values)):
        x = margin_l + idx * (bar_w + bar_gap)
        h = 0 if max_v == 0 else plot_h * value / max_v
        y = margin_t + plot_h - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{y - 5:.1f}" text-anchor="middle" font-family="Arial" font-size="11">{value:.3g}</text>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{margin_t + plot_h + 18}" text-anchor="end" transform="rotate(-45 {x + bar_w/2:.1f} {margin_t + plot_h + 18})" font-family="Arial" font-size="12">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_line_chart(
    path: Path,
    labels: list[str],
    series: list[tuple[str, list[float], str]],
    title: str,
    y_label: str,
    width: int = 1300,
    height: int = 620,
) -> None:
    margin_l, margin_r, margin_t, margin_b = 90, 170, 70, 120
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    all_values = [v for _, values, _ in series for v in values]
    max_v = max(all_values) if all_values else 1
    min_v = min(0, min(all_values) if all_values else 0)
    span = max(max_v - min_v, 1)
    n = max(1, len(labels))

    def xy(i: int, value: float) -> tuple[float, float]:
        x = margin_l + (plot_w * i / max(1, n - 1))
        y = margin_t + plot_h - (value - min_v) / span * plot_h
        return x, y

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-family="Arial" font-size="24" font-weight="700">{title}</text>',
        f'<text x="22" y="{margin_t + plot_h/2}" transform="rotate(-90 22 {margin_t + plot_h/2})" text-anchor="middle" font-family="Arial" font-size="15">{y_label}</text>',
        f'<line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" y2="{margin_t + plot_h}" stroke="#333"/>',
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" stroke="#333"/>',
    ]
    for i in range(6):
        y = margin_t + plot_h - plot_h * i / 5
        val = min_v + span * i / 5
        parts.append(f'<line x1="{margin_l}" y1="{y:.1f}" x2="{margin_l + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{margin_l - 8}" y="{y+5:.1f}" text-anchor="end" font-family="Arial" font-size="12">{val:.2g}</text>')
    for name, values, color in series:
        points = [xy(i, v) for i, v in enumerate(values)]
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{d}"/>')
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="{color}"/>')
    step = max(1, len(labels) // 18)
    for i, label in enumerate(labels):
        if i % step != 0 and i != len(labels) - 1:
            continue
        x, _ = xy(i, min_v)
        parts.append(f'<text x="{x:.1f}" y="{margin_t + plot_h + 18}" text-anchor="end" transform="rotate(-45 {x:.1f} {margin_t + plot_h + 18})" font-family="Arial" font-size="11">{label}</text>')
    for j, (name, _, color) in enumerate(series):
        y = margin_t + 20 + j * 24
        parts.append(f'<rect x="{margin_l + plot_w + 25}" y="{y-12}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{margin_l + plot_w + 45}" y="{y}" font-family="Arial" font-size="13">{name}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    mx, my = mean(xs), mean(ys)
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sx * sy)


def main() -> None:
    ensure_dirs()
    channels, objects = load_dicts()

    file_rows: list[dict[str, object]] = []
    year_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    duplicate_rows: list[dict[str, object]] = []

    month_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    day_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    hour_stats: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    weekday_stats: dict[int, list[int]] = defaultdict(lambda: [0, 0])

    system_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    sensor_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    object_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    value_kind_by_year: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])

    channel_stats: dict[str, ChannelStats] = defaultdict(ChannelStats)
    log_channels: set[str] = set()
    log_objects: set[str] = set()
    unknown_channels_counter: Counter[str] = Counter()
    top_text_values: Counter[str] = Counter()
    date_cache: dict[str, tuple[int, str, int, int] | None] = {}

    archives = sorted(RAW.glob("ext-journal-*.7z"))
    for archive in archives:
        year = archive.stem.split("-")[-1]
        size_mb = archive.stat().st_size / (1024 * 1024)
        print(f"[EDA] processing {archive.name} ({size_mb:.1f} MB)")

        proc, reader = open_archive_csv(archive)
        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}
        year_total = 0
        year_alarm = 0
        year_channels: set[str] = set()
        miss = Counter()
        invalid_alarm = 0
        invalid_dt = 0
        numeric_count = 0
        text_count = 0
        missing_value = 0
        negative_count = 0
        sentinel_count = 0
        num_stats = NumStats()
        first_date: str | None = None
        last_date: str | None = None
        hll = HyperLogLog()
        sample_ids: set[str] = set()
        sample_dup = 0
        sample_limit = 1_000_000
        malformed_rows = 0

        for row in reader:
            if len(row) < len(header):
                malformed_rows += 1
                continue
            year_total += 1
            event_id = row[idx["ид_события"]].strip()
            channel_id = row[idx["ид_канала_данных"]].strip()
            d = row[idx["дата"]].strip()
            t = row[idx["время"]].strip()
            alarm_raw = row[idx["тревожное"]]
            value_raw = row[idx["значение_датчика"]]

            for col_name, col_i in idx.items():
                if row[col_i].strip() == "":
                    miss[col_name] += 1

            if event_id.isdigit():
                hll.add_int(int(event_id))
                if year_total <= sample_limit:
                    if event_id in sample_ids:
                        sample_dup += 1
                    else:
                        sample_ids.add(event_id)

            alarm, alarm_status = parse_alarm(alarm_raw)
            if alarm_status != "ok":
                invalid_alarm += 1
            is_alarm = bool(alarm)
            if is_alarm:
                year_alarm += 1

            log_channels.add(channel_id)
            year_channels.add(channel_id)
            ch_info = channels.get(channel_id)
            if ch_info is None:
                unknown_channels_counter[channel_id] += 1
                system_name = "Нет в справочнике"
                sensor_name = "Нет в справочнике"
                object_id = "Нет в справочнике"
            else:
                system_name = ch_info.get("тип_инж_системы", "") or "Не указан"
                sensor_name = ch_info.get("тип_датчика", "") or "Не указан"
                object_id = ch_info.get("ид_объект", "") or "Не указан"
                log_objects.add(object_id)

            system_stats[system_name][0] += 1
            sensor_stats[sensor_name][0] += 1
            object_stats[object_id][0] += 1
            if is_alarm:
                system_stats[system_name][1] += 1
                sensor_stats[sensor_name][1] += 1
                object_stats[object_id][1] += 1

            if d not in date_cache:
                try:
                    dd = date.fromisoformat(d)
                    date_cache[d] = (dd.toordinal(), d[:7], dd.weekday(), dd.year)
                except ValueError:
                    date_cache[d] = None
            date_info = date_cache[d]
            if date_info is None:
                invalid_dt += 1
            else:
                ord_d, month, weekday, _ = date_info
                first_date = d if first_date is None else min(first_date, d)
                last_date = d if last_date is None else max(last_date, d)
                month_stats[month][0] += 1
                day_stats[d][0] += 1
                weekday_stats[weekday][0] += 1
                if is_alarm:
                    month_stats[month][1] += 1
                    day_stats[d][1] += 1
                    weekday_stats[weekday][1] += 1
                try:
                    hour = int(t[:2])
                    if 0 <= hour <= 23:
                        hour_stats[hour][0] += 1
                        if is_alarm:
                            hour_stats[hour][1] += 1
                except ValueError:
                    invalid_dt += 1

            ch = channel_stats[channel_id]
            ch.rows += 1
            ch.alarms += int(is_alarm)
            if date_info is not None:
                ord_d = date_info[0]
                ch.first_ord = ord_d if ch.first_ord is None else min(ch.first_ord, ord_d)
                ch.last_ord = ord_d if ch.last_ord is None else max(ch.last_ord, ord_d)

            value = parse_float(value_raw)
            if value_raw.strip() == "":
                missing_value += 1
                ch.missing_value += 1
                value_kind_by_year[year][2] += 1
            elif value is None:
                text_count += 1
                ch.text += 1
                value_kind_by_year[year][1] += 1
                if len(top_text_values) < 1000 or value_raw in top_text_values:
                    top_text_values[value_raw] += 1
            else:
                numeric_count += 1
                ch.numeric += 1
                ch.value_stats.add(value)
                num_stats.add(value)
                value_kind_by_year[year][0] += 1
                if value < 0:
                    negative_count += 1
                    ch.negative += 1
                if value in SENTINELS:
                    sentinel_count += 1
                    ch.sentinel += 1

        proc.wait()
        if proc.returncode != 0:
            stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
            raise RuntimeError(f"bsdtar failed for {archive}: {stderr}")

        distinct_est = hll.estimate()
        duplicate_est = max(0.0, year_total - distinct_est)
        year_rows.append(
            {
                "год": year,
                "строк": year_total,
                "тревог": year_alarm,
                "доля_тревог_%": pct(year_alarm, year_total),
                "уникальных_каналов": len(year_channels),
                "первая_дата": first_date,
                "последняя_дата": last_date,
                "числовых_значений": numeric_count,
                "текстовых_значений": text_count,
                "пустых_значений": missing_value,
                "отрицательных_чисел": negative_count,
                "sentinel_значений": sentinel_count,
                "минимум_значения": num_stats.min_value,
                "максимум_значения": num_stats.max_value,
                "среднее_значение": num_stats.avg,
                "std_значения": num_stats.std,
                "некорректных_дат_или_времени": invalid_dt,
                "некорректных_alarm": invalid_alarm,
                "битых_строк": malformed_rows,
            }
        )
        duplicate_rows.append(
            {
                "год": year,
                "строк": year_total,
                "примерная_уникальность_event_id": round(distinct_est),
                "примерная_оценка_дублей_event_id": round(duplicate_est),
                "точные_дубли_event_id_в_первых_1млн": sample_dup,
                "метод": "HyperLogLog по всем строкам + exact sample first 1M",
            }
        )
        for col in header:
            missing_rows.append({"файл": archive.name, "колонка": col, "пропусков": miss[col], "доля_пропусков_%": pct(miss[col], year_total)})
        file_rows.append({"файл": archive.name, "внутренний_csv": archive.with_suffix(".csv").name, "размер_mb": size_mb, "строк_с_заголовком": year_total + 1})

    # Dictionaries and coverage
    channels_in_dict = set(channels)
    objects_in_dict = set(objects)
    channels_missing_in_dict = sorted(log_channels - channels_in_dict)
    dict_channels_without_events = sorted(channels_in_dict - log_channels)
    objects_missing_in_dict = sorted(log_objects - objects_in_dict)
    dict_objects_without_events = sorted(objects_in_dict - log_objects)
    coverage_rows = [
        {"проверка": "Каналы из журналов, всего", "значение": len(log_channels)},
        {"проверка": "Каналы в справочнике, всего", "значение": len(channels_in_dict)},
        {"проверка": "Каналы журнала без справочника", "значение": len(channels_missing_in_dict)},
        {"проверка": "Каналы справочника без событий", "значение": len(dict_channels_without_events)},
        {"проверка": "Объекты из событий через каналы", "значение": len(log_objects)},
        {"проверка": "Объекты в справочнике", "значение": len(objects_in_dict)},
        {"проверка": "Объекты событий без справочника", "значение": len(objects_missing_in_dict)},
        {"проверка": "Объекты справочника без событий", "значение": len(dict_objects_without_events)},
    ]

    # Aggregation tables
    month_rows = [
        {"месяц": m, "строк": v[0], "тревог": v[1], "доля_тревог_%": pct(v[1], v[0])}
        for m, v in sorted(month_stats.items())
    ]
    hour_rows = [
        {"час": h, "строк": hour_stats[h][0], "тревог": hour_stats[h][1], "доля_тревог_%": pct(hour_stats[h][1], hour_stats[h][0])}
        for h in range(24)
    ]
    weekday_rows = [
        {"день_недели": WEEKDAYS_RU[i], "строк": weekday_stats[i][0], "тревог": weekday_stats[i][1], "доля_тревог_%": pct(weekday_stats[i][1], weekday_stats[i][0])}
        for i in range(7)
    ]

    def summary_rows_from_counter(stats: dict[str, list[int]], key_name: str) -> list[dict[str, object]]:
        rows = []
        total_rows = sum(v[0] for v in stats.values())
        for key, (rows_count, alarms_count) in stats.items():
            rows.append(
                {
                    key_name: key,
                    "строк": rows_count,
                    "доля_строк_%": pct(rows_count, total_rows),
                    "тревог": alarms_count,
                    "доля_тревог_%": pct(alarms_count, rows_count),
                }
            )
        return sorted(rows, key=lambda r: (-int(r["строк"]), str(r[key_name])))

    system_rows = summary_rows_from_counter(system_stats, "тип_инж_системы")
    sensor_rows = summary_rows_from_counter(sensor_stats, "тип_датчика")
    object_rows = summary_rows_from_counter(object_stats, "ид_объект")

    channel_rows = []
    for channel_id, st in channel_stats.items():
        ch_info = channels.get(channel_id, {})
        active_days = (st.last_ord - st.first_ord + 1) if st.first_ord and st.last_ord else None
        channel_rows.append(
            {
                "ид_канала_данных": channel_id,
                "тип_инж_системы": ch_info.get("тип_инж_системы", "Нет в справочнике"),
                "тип_датчика": ch_info.get("тип_датчика", "Нет в справочнике"),
                "ид_объект": ch_info.get("ид_объект", "Нет в справочнике"),
                "строк": st.rows,
                "тревог": st.alarms,
                "доля_тревог_%": pct(st.alarms, st.rows),
                "числовых_значений": st.numeric,
                "доля_числовых_%": pct(st.numeric, st.rows),
                "текстовых_значений": st.text,
                "пустых_значений": st.missing_value,
                "отрицательных_чисел": st.negative,
                "sentinel_значений": st.sentinel,
                "активных_дней": active_days,
                "событий_в_день": st.rows / active_days if active_days else None,
                "среднее_значение": st.value_stats.avg,
                "std_значения": st.value_stats.std,
                "минимум_значения": st.value_stats.min_value,
                "максимум_значения": st.value_stats.max_value,
            }
        )
    channel_top_activity = sorted(channel_rows, key=lambda r: -int(r["строк"]))[:50]
    channel_top_alarm_rate = sorted(
        [r for r in channel_rows if int(r["строк"]) >= 1000],
        key=lambda r: (-float(r["доля_тревог_%"]), -int(r["строк"])),
    )[:50]

    value_kind_rows = [
        {
            "год": year,
            "числовые": vals[0],
            "текстовые": vals[1],
            "пустые": vals[2],
            "доля_числовых_%": pct(vals[0], sum(vals)),
            "доля_текстовых_%": pct(vals[1], sum(vals)),
            "доля_пустых_%": pct(vals[2], sum(vals)),
        }
        for year, vals in sorted(value_kind_by_year.items())
    ]
    text_value_rows = [{"значение": k, "строк": v} for k, v in top_text_values.most_common(50)]

    # Channel-level correlations.
    numeric_features: dict[str, list[float]] = defaultdict(list)
    target_alarm_rate: list[float] = []
    for r in channel_rows:
        if int(r["строк"]) < 100:
            continue
        target_alarm_rate.append(float(r["доля_тревог_%"]))
        for feature in [
            "строк",
            "числовых_значений",
            "доля_числовых_%",
            "текстовых_значений",
            "пустых_значений",
            "отрицательных_чисел",
            "sentinel_значений",
            "активных_дней",
            "событий_в_день",
            "среднее_значение",
            "std_значения",
            "минимум_значения",
            "максимум_значения",
        ]:
            value = r.get(feature)
            numeric_features[feature].append(float(value) if value not in (None, "") else 0.0)
    corr_target_rows = []
    for feature, values in numeric_features.items():
        corr = pearson(values, target_alarm_rate)
        corr_target_rows.append({"feature": feature, "target": "доля_тревог_%", "pearson": corr})
    corr_target_rows = sorted(corr_target_rows, key=lambda r: -abs(float(r["pearson"] or 0)))

    feature_names = list(numeric_features)
    corr_matrix_rows = []
    for f1 in feature_names:
        row = {"feature": f1}
        for f2 in feature_names:
            row[f2] = pearson(numeric_features[f1], numeric_features[f2])
        corr_matrix_rows.append(row)

    data_quality_rows = [
        {"проверка": "Всего строк в годовых журналах", "значение": sum(int(r["строк"]) for r in year_rows), "комментарий": "Без строки заголовка"},
        {"проверка": "Всего тревог", "значение": sum(int(r["тревог"]) for r in year_rows), "комментарий": "По полю тревожное"},
        {"проверка": "Общая доля тревог, %", "значение": pct(sum(int(r["тревог"]) for r in year_rows), sum(int(r["строк"]) for r in year_rows)), "комментарий": "Сильный дисбаланс классов"},
        {"проверка": "Каналы без справочника", "значение": len(channels_missing_in_dict), "комментарий": "Влияет на построение признаков по типу датчика и объекту"},
        {"проверка": "Объекты без справочника", "значение": len(objects_missing_in_dict), "комментарий": "Проверка связки канал -> объект"},
        {"проверка": "2021 год", "значение": next((r["доля_тревог_%"] for r in year_rows if r["год"] == "2021"), None), "комментарий": "Год нужно проверять отдельно из-за внедрения новой версии мониторинга"},
        {"проверка": "Target", "значение": "неоднозначен", "комментарий": "Поле тревожное есть, но нет подтверждения ОДС/ложности/истинности инцидента"},
    ]

    # Write tables.
    write_csv(TABLES / "file_inventory.csv", file_rows)
    write_csv(TABLES / "year_summary.csv", year_rows)
    write_csv(TABLES / "missing_by_file.csv", missing_rows)
    write_csv(TABLES / "duplicate_estimates.csv", duplicate_rows)
    write_csv(TABLES / "dictionary_coverage.csv", coverage_rows)
    write_csv(TABLES / "month_summary.csv", month_rows)
    write_csv(TABLES / "hour_summary.csv", hour_rows)
    write_csv(TABLES / "weekday_summary.csv", weekday_rows)
    write_csv(TABLES / "engineering_system_summary.csv", system_rows)
    write_csv(TABLES / "sensor_type_summary.csv", sensor_rows)
    write_csv(TABLES / "object_summary.csv", object_rows)
    write_csv(TABLES / "channel_top_activity.csv", channel_top_activity)
    write_csv(TABLES / "channel_top_alarm_rate.csv", channel_top_alarm_rate)
    write_csv(TABLES / "value_kind_by_year.csv", value_kind_rows)
    write_csv(TABLES / "top_text_values.csv", text_value_rows)
    write_csv(TABLES / "target_correlation_channel_features.csv", corr_target_rows)
    write_csv(TABLES / "feature_correlation_matrix.csv", corr_matrix_rows)
    write_csv(TABLES / "data_quality_checks.csv", data_quality_rows)
    write_csv(TABLES / "unknown_channels_top.csv", [{"ид_канала_данных": k, "строк": v} for k, v in unknown_channels_counter.most_common(100)])

    # Raw stats JSON.
    raw = {
        "year_summary": year_rows,
        "coverage": coverage_rows,
        "duplicate_estimates": duplicate_rows,
        "top_text_values": text_value_rows[:20],
    }
    (RAW_STATS / "eda_stats.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    # Figures.
    svg_bar_chart(
        FIGURES / "events_by_year.svg",
        [str(r["год"]) for r in year_rows],
        [float(r["строк"]) / 1_000_000 for r in year_rows],
        "Количество событий по годам",
        "млн строк",
        "#2563eb",
    )
    svg_bar_chart(
        FIGURES / "alarm_rate_by_year.svg",
        [str(r["год"]) for r in year_rows],
        [float(r["доля_тревог_%"]) for r in year_rows],
        "Доля тревожных событий по годам",
        "доля тревог, %",
        "#dc2626",
    )
    svg_line_chart(
        FIGURES / "alarm_rate_by_month.svg",
        [str(r["месяц"]) for r in month_rows],
        [("Доля тревог, %", [float(r["доля_тревог_%"]) for r in month_rows], "#dc2626")],
        "Доля тревожных событий по месяцам",
        "доля тревог, %",
    )
    svg_bar_chart(
        FIGURES / "alarm_rate_by_hour.svg",
        [str(r["час"]) for r in hour_rows],
        [float(r["доля_тревог_%"]) for r in hour_rows],
        "Доля тревожных событий по часу суток",
        "доля тревог, %",
        "#7c3aed",
    )
    svg_bar_chart(
        FIGURES / "events_by_engineering_system.svg",
        [str(r["тип_инж_системы"]) for r in system_rows[:10]],
        [float(r["строк"]) / 1_000_000 for r in system_rows[:10]],
        "События по типам инженерных систем",
        "млн строк",
        "#059669",
    )
    svg_bar_chart(
        FIGURES / "alarm_rate_by_engineering_system.svg",
        [str(r["тип_инж_системы"]) for r in system_rows[:10]],
        [float(r["доля_тревог_%"]) for r in system_rows[:10]],
        "Доля тревог по типам инженерных систем",
        "доля тревог, %",
        "#dc2626",
    )
    svg_bar_chart(
        FIGURES / "top_sensor_types_by_events.svg",
        [str(r["тип_датчика"]) for r in sensor_rows[:15]],
        [float(r["строк"]) / 1_000_000 for r in sensor_rows[:15]],
        "Топ типов датчиков по числу событий",
        "млн строк",
        "#0891b2",
    )
    svg_bar_chart(
        FIGURES / "value_kind_by_year_numeric_share.svg",
        [str(r["год"]) for r in value_kind_rows],
        [float(r["доля_числовых_%"]) for r in value_kind_rows],
        "Доля числовых значений датчика по годам",
        "доля числовых, %",
        "#9333ea",
    )
    svg_bar_chart(
        FIGURES / "top_channels_by_activity.svg",
        [str(r["ид_канала_данных"]) for r in channel_top_activity[:20]],
        [float(r["строк"]) / 1_000_000 for r in channel_top_activity[:20]],
        "Топ-20 каналов по активности",
        "млн строк",
        "#ea580c",
    )

    # Report.
    total_rows = sum(int(r["строк"]) for r in year_rows)
    total_alarms = sum(int(r["тревог"]) for r in year_rows)
    max_alarm_year = max(year_rows, key=lambda r: float(r["доля_тревог_%"]))
    max_rows_year = max(year_rows, key=lambda r: int(r["строк"]))
    top_system_alarm = max(system_rows, key=lambda r: float(r["доля_тревог_%"]) if int(r["строк"]) > 1000 else -1)
    top_corr = corr_target_rows[0] if corr_target_rows else {}

    report = f"""# EDA исходных данных ЛЦТ: инженерные коллекторы

## Краткое резюме

Полный потоковый анализ годовых журналов охватил {total_rows:,} строк за 2019-2026 годы. Тревожных событий найдено {total_alarms:,}, общая доля тревог составляет {pct(total_alarms, total_rows):.4f}%. Это означает сильный дисбаланс классов: даже простая постановка target по полю `тревожное` будет требовать аккуратной валидации, подбора порогов и метрик Precision/Recall.

Самый объемный год по числу событий - {max_rows_year['год']} ({int(max_rows_year['строк']):,} строк). Самая высокая доля тревог по годам - {max_alarm_year['год']} ({float(max_alarm_year['доля_тревог_%']):.4f}%). Это важно сверить с Q&A по кейсу: 2021 год мог быть нетипичным из-за внедрения новой версии системы мониторинга.

Главный вывод по target: поле `тревожное` можно использовать как базовый прокси-target для тревожного события, но оно не является однозначной разметкой отказа датчика, аварии или подтвержденного инцидента. В данных нет полной связки с решением диспетчера, ложностью/истинностью тревоги и журналом ОДС.

## Файлы и объемы

{markdown_table(file_rows, ['файл', 'размер_mb', 'строк_с_заголовком'])}

## Сводка по годам

{markdown_table(year_rows, ['год', 'строк', 'тревог', 'доля_тревог_%', 'уникальных_каналов', 'первая_дата', 'последняя_дата', 'числовых_значений', 'текстовых_значений'])}

![Количество событий по годам](figures/events_by_year.svg)

![Доля тревожных событий по годам](figures/alarm_rate_by_year.svg)

## Покрытие справочников

{markdown_table(coverage_rows, ['проверка', 'значение'])}

Связка каналов с объектами в локальном справочнике присутствует через `ид_объект`. Если канал есть в журнале, но отсутствует в справочнике, для него нельзя напрямую получить тип датчика, тип инженерной системы и объект. Такие случаи нужно обрабатывать отдельной категорией.

## Качество данных

{markdown_table(data_quality_rows, ['проверка', 'значение', 'комментарий'])}

Проверка дублей по всем архивам выполнена масштабируемо: точная проверка первых 1 млн строк каждого года и приблизительная оценка уникальности `ид_события` через HyperLogLog по всем строкам. Полная точная проверка всех ID потребовала бы хранить сотни миллионов идентификаторов или выполнять тяжелую внешнюю сортировку.

{markdown_table(duplicate_rows, ['год', 'строк', 'примерная_уникальность_event_id', 'примерная_оценка_дублей_event_id', 'точные_дубли_event_id_в_первых_1млн'])}

## Значения датчиков

В поле `значение_датчика` смешаны числовые показания и текстовые состояния. Это принципиально важно: температурные и газовые датчики можно анализировать как числовые ряды, а бинарные/дискретные каналы нужно анализировать через события, состояния и паттерны переключений.

{markdown_table(value_kind_rows, ['год', 'числовые', 'текстовые', 'пустые', 'доля_числовых_%', 'доля_текстовых_%'])}

![Доля числовых значений датчика по годам](figures/value_kind_by_year_numeric_share.svg)

Топ текстовых значений:

{markdown_table(text_value_rows, ['значение', 'строк'], limit=20)}

## Временные паттерны

![Доля тревожных событий по месяцам](figures/alarm_rate_by_month.svg)

![Доля тревожных событий по часу суток](figures/alarm_rate_by_hour.svg)

## Инженерные системы и типы датчиков

{markdown_table(system_rows[:12], ['тип_инж_системы', 'строк', 'доля_строк_%', 'тревог', 'доля_тревог_%'])}

![События по типам инженерных систем](figures/events_by_engineering_system.svg)

![Доля тревог по типам инженерных систем](figures/alarm_rate_by_engineering_system.svg)

Топ типов датчиков по числу событий:

{markdown_table(sensor_rows[:20], ['тип_датчика', 'строк', 'доля_строк_%', 'тревог', 'доля_тревог_%'])}

![Топ типов датчиков по числу событий](figures/top_sensor_types_by_events.svg)

## Активность каналов

События распределены по каналам неравномерно: есть каналы с очень высокой активностью. Это может быть как нормальная особенность телеметрии, так и признак каналов, которые будут доминировать в обучении модели, если не делать нормализацию или агрегацию по временным окнам.

{markdown_table(channel_top_activity[:20], ['ид_канала_данных', 'тип_инж_системы', 'тип_датчика', 'ид_объект', 'строк', 'тревог', 'доля_тревог_%'])}

![Топ-20 каналов по активности](figures/top_channels_by_activity.svg)

Каналы с высокой долей тревог среди каналов с минимум 1000 событий:

{markdown_table(channel_top_alarm_rate[:20], ['ид_канала_данных', 'тип_инж_системы', 'тип_датчика', 'ид_объект', 'строк', 'тревог', 'доля_тревог_%'])}

## Корреляции с прокси-target

Корреляции посчитаны на уровне каналов: target - доля тревожных событий канала, признаки - агрегаты активности и значений канала. Это не заменяет будущую временную ML-валидацию, но помогает увидеть первичные связи.

{markdown_table(corr_target_rows, ['feature', 'target', 'pearson'])}

Самая заметная первичная связь с долей тревог: `{top_corr.get('feature', '')}` с Pearson {fmt(top_corr.get('pearson'))}. Интерпретацию нужно делать осторожно: корреляция на уровне каналов не означает причинность и может отражать тип датчика или режим записи данных.

## Однозначность target

Target не определяется однозначно. Поле `тревожное` говорит о тревожном событии в мониторинге, но не говорит, было ли это подтвержденной аварией, отказом датчика, ложным срабатыванием или действием человека. Из Q&A известно, что журнал ОДС и helpdesk не имеют прямой полной связки с каждым событием датчика. Поэтому для MVP разумно рассматривать несколько постановок:

1. `alarm_event` - текущее событие является тревожным по полю `тревожное`.
2. `alarm_next_24h` - по каналу или объекту появится тревога в следующие 24 часа.
3. `alarm_burst_next_24h` - появится серия тревог или аномально высокая частота событий.
4. `sensor_anomaly` - значение или паттерн канала выглядит нетипично относительно истории.

Для хакатона наиболее безопасный первый target - прогноз тревожного события или тревожного кластера на горизонте 24 часа. При этом в документации обязательно нужно написать, что это прокси-target, а не подтвержденная авария.

## Нетривиальные особенности данных

- Датасет очень большой: более {total_rows:,} строк, поэтому EDA и feature engineering должны быть потоковыми или батчевыми.
- Значения датчиков смешивают числа и текстовые состояния; единая обработка `значение_датчика` как float невозможна.
- 2021 год требует отдельной проверки как потенциально аномальный период внедрения новой системы мониторинга.
- Активность каналов сильно неравномерна: часть каналов генерирует непропорционально много событий.
- Target по полю `тревожное` есть, но бизнес-target отказа/аварии не размечен однозначно.

## Сохраненные таблицы

- `tables/file_inventory.csv`
- `tables/year_summary.csv`
- `tables/missing_by_file.csv`
- `tables/duplicate_estimates.csv`
- `tables/dictionary_coverage.csv`
- `tables/month_summary.csv`
- `tables/hour_summary.csv`
- `tables/weekday_summary.csv`
- `tables/engineering_system_summary.csv`
- `tables/sensor_type_summary.csv`
- `tables/object_summary.csv`
- `tables/channel_top_activity.csv`
- `tables/channel_top_alarm_rate.csv`
- `tables/value_kind_by_year.csv`
- `tables/top_text_values.csv`
- `tables/target_correlation_channel_features.csv`
- `tables/feature_correlation_matrix.csv`
- `tables/data_quality_checks.csv`
"""
    (REPORT / "eda_report.md").write_text(report, encoding="utf-8")
    print(f"[EDA] report written to {REPORT / 'eda_report.md'}")


if __name__ == "__main__":
    main()
