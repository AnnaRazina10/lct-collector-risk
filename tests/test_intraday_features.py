"""Guard event-time ordering, chunk reduction, and daily forecast cutoffs."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/modeling"))
from intraday_features import RAW_COLUMNS, combine_parts, summarize_chunk


def events(rows):
    return pd.DataFrame(rows, columns=RAW_COLUMNS, dtype=str)


def aggregate(raw, chunk_size=None):
    chunk_size = chunk_size or len(raw)
    parts = [summarize_chunk(raw.iloc[start:start + chunk_size], start)[0]
             for start in range(0, len(raw), chunk_size)]
    return combine_parts(parts)


def day_row(frame, channel="a", date="2025-01-01"):
    return frame.loc[frame.channel_id.eq(channel) & frame.date.eq(pd.Timestamp(date))].iloc[0]


class IntradayFeaturesTest(unittest.TestCase):
    def test_first_and_last_use_event_time_not_file_order(self):
        raw = events([
            ["30", "a", "2025-01-01", "20:00:00", "false"],
            ["10", "a", "2025-01-01", "06:00:00", "true"],
            ["20", "a", "2025-01-01", "18:30:00", "true"],
        ])
        row = day_row(aggregate(raw))
        self.assertEqual(row.intra_first_event_minute, 360)
        self.assertEqual(row.intra_first_event_is_alarm, 1)
        self.assertEqual(row.intra_last_event_minute, 1200)
        self.assertEqual(row.intra_last_event_is_alarm, 0)
        self.assertEqual(row.intra_last_alarm_minute, 1110)
        self.assertEqual(row.intra_alarm_count_last6h, 1)
        self.assertEqual(row.intra_events_last6h, 2)

    def test_equal_timestamps_use_numeric_id_then_original_row_order(self):
        raw = events([
            ["10", "a", "2025-01-01", "20:00:00", "true"],
            ["2", "a", "2025-01-01", "20:00:00", "false"],
            ["9", "a", "2025-01-01", "20:00:00", "false"],
        ])
        row = day_row(aggregate(raw, chunk_size=1))
        self.assertEqual(row.intra_first_event_is_alarm, 0)
        self.assertEqual(row.intra_last_event_is_alarm, 1)
        self.assertEqual(row.intra_last_timestamp_ties, 2)

        # A later record with the same timestamp AND id wins the final tie.
        repeated = pd.concat([raw, events([
            ["10", "a", "2025-01-01", "20:00:00", "false"],
        ])], ignore_index=True)
        row = day_row(aggregate(repeated, chunk_size=2))
        self.assertEqual(row.intra_last_event_is_alarm, 0)
        self.assertEqual(row.intra_last_timestamp_ties, 3)

    def test_reduction_is_independent_of_chunk_boundaries(self):
        raw = events([
            ["1", "a", "2025-01-01", "00:00:00", "true"],
            ["6", "b", "2025-01-01", "22:00:00", "false"],
            ["2", "a", "2025-01-01", "18:00:00", "true"],
            ["10", "a", "2025-01-01", "23:59:59", "false"],
            ["8", "a", "2025-01-01", "23:59:59", "true"],
            ["5", "b", "2025-01-01", "12:00:00", "true"],
            ["11", "a", "2025-01-02", "12:00:00", "false"],
            ["10", "a", "2025-01-01", "23:59:59", "true"],
        ])
        expected = aggregate(raw)
        for chunk_size in [1, 2, 3, 5]:
            with self.subTest(chunk_size=chunk_size):
                pd.testing.assert_frame_equal(expected, aggregate(raw, chunk_size))
        self.assertEqual(day_row(expected).intra_last_timestamp_ties, 2)
        self.assertEqual(day_row(expected, "b").intra_last_timestamp_ties, 0)

    def test_last_six_hours_include_1800_and_exclude_next_midnight(self):
        raw = events([
            ["1", "a", "2025-01-01", "11:59:59", "true"],
            ["2", "a", "2025-01-01", "12:00:00", "true"],
            ["3", "a", "2025-01-01", "17:59:59", "true"],
            ["4", "a", "2025-01-01", "18:00:00", "true"],
            ["5", "a", "2025-01-01", "23:59:59", "false"],
            ["6", "a", "2025-01-02", "00:00:00", "true"],
        ])
        result = aggregate(raw, chunk_size=2)
        today = day_row(result)
        self.assertEqual(today.intra_alarm_count_last6h, 1)
        self.assertEqual(today.intra_alarm_count_last12h, 3)
        self.assertEqual(today.intra_events_last6h, 2)
        self.assertEqual(today.intra_last_alarm_minute, 1080)
        tomorrow = day_row(result, date="2025-01-02")
        self.assertEqual(tomorrow.intra_last_alarm_minute, 0)
        self.assertEqual(tomorrow.intra_alarm_count_last6h, 0)
        self.assertEqual(tomorrow.intra_alarm_count_last12h, 0)

    def test_future_and_other_channels_cannot_change_earlier_day(self):
        current = events([
            ["1", "a", "2025-01-01", "18:00:00", "false"],
            ["2", "a", "2025-01-01", "20:00:00", "false"],
        ])
        additional = events([
            ["3", "a", "2025-01-02", "01:00:00", "true"],
            ["4", "b", "2025-01-01", "23:59:59", "true"],
            ["5", "a", "2025-01-03", "23:59:59", "true"],
        ])
        before = aggregate(current)
        after = aggregate(pd.concat([current, additional], ignore_index=True), 2)
        after = after.loc[after.channel_id.eq("a") & after.date.eq(pd.Timestamp("2025-01-01"))]
        pd.testing.assert_frame_equal(before, after.reset_index(drop=True))
        # No alarm is distinct from an alarm exactly at midnight.
        self.assertEqual(day_row(before).intra_last_alarm_minute, -1)
        self.assertEqual(day_row(before).intra_last_event_is_alarm, 0)


if __name__ == "__main__":
    unittest.main()
