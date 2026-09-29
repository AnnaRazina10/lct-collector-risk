"""Causality guards of the archive exporter, without loading models or raw data."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/modeling"))
import export_object_demo as exporter


class ObjectExportTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({"date": pd.to_datetime(["2026-06-27", "2026-06-28"]),
            "object_id": [1, 1], "target_dayofweek": [0, 1], "observed_today": [1, 1],
            exporter.source.TARGET: [np.nan, np.nan]})

    def test_window_starts_24_hours_after_issue_and_excludes_feature_day(self):
        times = exporter.forecast_times("2026-06-28")
        self.assertEqual(times["issue_time"], "2026-06-29T00:00:00+03:00")
        self.assertEqual(times["forecast_start"], "2026-06-30T00:00:00+03:00")
        self.assertEqual(times["forecast_end"], "2026-07-01T00:00:00+03:00")
        self.assertEqual(times["feature_cutoff"], "2026-06-28T23:59:59+03:00")

    def test_history_is_explicitly_truncated_before_inference(self):
        with patch.object(exporter.source, "prepare", return_value=(self.frame(), [], {})) as prepare:
            frame, _ = exporter.prediction_frame("2026-06-28", {"features": ["target_dayofweek"]})
        prepare.assert_called_once_with("2026-06-28")
        self.assertEqual(len(frame), 1)
        self.assertTrue(frame[exporter.source.TARGET].isna().all())
        self.assertEqual(frame.observed_days_7d.iloc[0], 2)

    def test_rejects_known_future_labels_and_wrong_calendar(self):
        for column, value in [(exporter.source.TARGET, 1), ("target_dayofweek", 3), ("date", pd.Timestamp("2026-06-29"))]:
            frame = self.frame()
            frame.loc[1, column] = value
            with self.subTest(column=column), patch.object(exporter.source, "prepare", return_value=(frame, [], {})):
                with self.assertRaises(ValueError):
                    exporter.prediction_frame("2026-06-28", {"features": ["target_dayofweek"]})

    def test_rejects_target_as_feature(self):
        with patch.object(exporter.source, "prepare", return_value=(self.frame(), [], {})):
            with self.assertRaises(ValueError):
                exporter.prediction_frame("2026-06-28", {"features": [exporter.source.TARGET]})


if __name__ == "__main__":
    unittest.main()
