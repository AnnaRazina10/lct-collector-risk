import sys
import unittest
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/modeling'))
from uncertainty_research_round import WeightedAP


class WeightedAPTests(unittest.TestCase):
    def test_weighted_ties_and_absent_days_match_sklearn(self):
        rng = np.random.default_rng(41)
        y = rng.integers(0, 2, 500)
        score = rng.integers(0, 12, 500) / 12
        days = rng.integers(0, 15, 500)
        metric = WeightedAP(y, score, days)
        for _ in range(10):
            weights = rng.integers(0, 6, 15)
            expected = average_precision_score(y, score, sample_weight=weights[days])
            self.assertAlmostEqual(metric(weights), expected, places=12)


if __name__ == '__main__':
    unittest.main()
