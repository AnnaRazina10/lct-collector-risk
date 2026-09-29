"""Guard temporal causality, channel isolation and operating-point accounting."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/modeling"))
import numpy as np
from improve_baseline import rolling_sum, days_since, metrics, choose_threshold


class TemporalFeaturesTest(unittest.TestCase):
    def test_trailing_windows_do_not_cross_channels(self):
        x = np.array([[0, 1, 2, 3], [10, 0, 0, 0]], dtype=float)
        np.testing.assert_equal(rolling_sum(x, 2), [[0, 1, 3, 5], [10, 10, 0, 0]])

    def test_future_changes_cannot_change_earlier_features(self):
        x = np.array([[0, 1, 0, 1, 0]], dtype=bool)
        changed = x.copy(); changed[:, 3:] = ~changed[:, 3:]
        for fn in [lambda a: rolling_sum(a, 3), days_since]:
            np.testing.assert_equal(fn(x)[:, :3], fn(changed)[:, :3])

    def test_no_history_is_not_recent_normal(self):
        np.testing.assert_equal(days_since(np.array([[0, 0, 1, 0]], dtype=bool)), [[-1, -1, 0, 1]])

    def test_threshold_and_confusion_counts(self):
        y = np.array([0, 1, 0, 1]); score = np.array([.1, .8, .7, .9])
        threshold, operating = choose_threshold(y, score)
        result = metrics(y, score, threshold)
        self.assertEqual(result["true_positive"], 2)
        self.assertEqual(result["false_positive"], 0)
        self.assertEqual(result["false_negative"], 0)
        self.assertTrue(operating["requirement_achievable"])


if __name__ == "__main__":
    unittest.main()
