import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/modeling'))
from object_statistical_candidate import markov_forecast, seasonal_forecast


class ObjectStatisticsTests(unittest.TestCase):
    def test_future_states_cannot_change_past_markov_forecasts(self):
        prior = np.full((3, 3), 1/3)
        states = np.tile([0, 1, 2, 1], 12)
        changed = states.copy(); changed[24:] = 2
        np.testing.assert_array_equal(markov_forecast(states, prior)[:24], markov_forecast(changed, prior)[:24])

    def test_markov_propagates_two_steps(self):
        prior = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
        self.assertEqual(markov_forecast(np.array([0]), prior)[0], 1)
        self.assertEqual(markov_forecast(np.array([1]), prior)[0], 0)

    def test_seasonal_forecast_targets_d_plus_two_without_future(self):
        prior = np.arange(7) / 7
        score = seasonal_forecast(np.array([1]), np.array([0]), prior)
        self.assertAlmostEqual(score[0], prior[2])
        alarms = np.tile([0, 1], 21); weekday = np.arange(42) % 7
        changed = alarms.copy(); changed[21:] = 1
        np.testing.assert_array_equal(seasonal_forecast(alarms, weekday, prior)[:21], seasonal_forecast(changed, weekday, prior)[:21])


if __name__ == '__main__':
    unittest.main()
