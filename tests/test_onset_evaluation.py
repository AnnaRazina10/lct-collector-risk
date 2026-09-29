"""Synthetic fixed-budget and paired-calendar-block regression tests."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/modeling"))
from onset_evaluation import daily_topk, paired_week_bootstrap, summarize


def population(days=14, start="2025-01-05"):
    dates = pd.date_range(start, periods=days)
    return pd.DataFrame({"object_id": np.tile(["A", "B"], days), "date": np.repeat(dates, 2),
                         "onset_target": np.tile([1, 0], days), "alarm_target": np.tile([1, 0], days)})


class OnsetEvaluationTests(unittest.TestCase):
    def test_ties_are_broken_by_string_id_and_ignore_row_order_and_labels(self):
        frame = pd.DataFrame({"object_id": [2, 10, "A"], "date": pd.to_datetime(["2025-01-05"]*3),
                              "onset_target": [1, 0, 0], "alarm_target": [1, 1, 1]})
        chosen = daily_topk(frame, [.5, .5, .5], k=1)
        self.assertEqual(chosen.tolist(), [False, True, False])  # "10" sorts before "2".
        self.assertEqual(chosen.dtype, bool)
        order = [2, 0, 1]
        shuffled = frame.iloc[order].copy()
        shuffled["onset_target"] = [1, 1, 0]
        self.assertEqual(daily_topk(shuffled, [.5]*3, k=1).tolist(), [False, False, True])

    def test_unequal_candidate_counts_keep_exact_budget_or_reject(self):
        frame = pd.DataFrame({"object_id": ["A", "B", "C", "A", "B", "C", "D"],
                              "date": pd.to_datetime(["2025-01-01"]*3+["2025-01-02"]*4),
                              "onset_target": [0]*7, "alarm_target": [0]*7})
        scores = [.8, .7, .1, .1, .2, .3, .4]
        selected = daily_topk(frame, scores, k=2)
        self.assertEqual(selected.tolist(), [True, True, False, False, False, True, True])
        self.assertEqual(frame.assign(selected=selected).groupby("date").selected.sum().tolist(), [2, 2])
        with self.assertRaisesRegex(ValueError, "at least k"):
            daily_topk(frame, scores, k=4)

    def test_known_metrics_and_exhaustive_warning_partition(self):
        frame = pd.DataFrame({"object_id": ["A", "B", "C"]*2,
                              "date": pd.to_datetime(["2025-01-01"]*3+["2025-01-02"]*3),
                              "onset_target": [1, 0, 0, 0, 1, 1], "alarm_target": [1, 1, 0, 0, 1, 1]})
        result = summarize(frame, [.9, .8, .1, .9, .8, .1], k=2)
        self.assertEqual(result["warning_partition"], {"onset": 2, "repeat_alarm": 1, "no_alarm": 1})
        self.assertEqual(result["warnings"], 4)
        self.assertEqual(result["warnings_per_day"], 2)
        self.assertEqual(result["tp"], 2)
        self.assertEqual(result["precision_at_k"], .5)
        self.assertAlmostEqual(result["recall"], 2/3)
        self.assertAlmostEqual(result["average_precision"], .5)

    def test_missing_nonbinary_inconsistent_and_duplicate_labels_rejected(self):
        frame = population(days=2)
        bad = []
        missing = frame.copy(); missing.loc[0, "onset_target"] = np.nan; bad.append(missing)
        nonbinary = frame.copy(); nonbinary.loc[0, "onset_target"] = 2; bad.append(nonbinary)
        string_labels = frame.copy(); string_labels["onset_target"] = string_labels.onset_target.astype(str); bad.append(string_labels)
        contradictory = frame.copy(); contradictory.loc[0, "alarm_target"] = 0; bad.append(contradictory)
        duplicate = frame.copy(); duplicate.loc[1, "object_id"] = "A"; bad.append(duplicate)
        blank = frame.copy(); blank.loc[0, "object_id"] = " "; bad.append(blank)
        not_day = frame.copy(); not_day.loc[0, "date"] += pd.Timedelta(hours=1); bad.append(not_day)
        for i, value in enumerate(bad):
            with self.subTest(i=i), self.assertRaises(ValueError):
                daily_topk(value, [0]*4, k=1)
        with self.assertRaises(ValueError):
            summarize(frame.drop(columns="alarm_target"), [0]*4, k=1)

    def test_invalid_scores_and_budgets_rejected(self):
        frame = population(days=2)
        for scores in [[0, np.nan, 0, 0], [0, np.inf, 0, 0], [0]*3, [[0, 0], [0, 0]], ["invalid"]*4]:
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                summarize(frame, scores, k=1)
        for k in [0, -1, True, 1.5]:
            with self.subTest(k=k), self.assertRaises(ValueError):
                daily_topk(frame, [0]*4, k=k)

    def test_no_positive_labels_has_defined_zero_ap_but_undefined_recall(self):
        frame = population()
        frame["onset_target"] = 0
        result = summarize(frame, [0]*len(frame), k=1)
        self.assertEqual(result["average_precision"], 0)
        self.assertEqual(result["precision_at_k"], 0)
        self.assertIsNone(result["recall"])
        self.assertTrue(result["undefined_recall"])
        paired = paired_week_bootstrap(frame, [0]*len(frame), [1]*len(frame), k=1, n=40)
        self.assertIsNone(paired["delta_recall"]["ci95"])
        self.assertEqual(paired["recall_replicates_without_onsets"], 40)
        json.dumps(paired, allow_nan=False)

    def test_paired_identical_models_have_exactly_zero_intervals_with_varying_weeks(self):
        frame = population(days=21)
        # Different success rates per week; unpaired resampling would add noise.
        scores = np.r_[np.tile([1, 0], 7), np.tile([0, 1], 7), np.tile([1, 0], 7)]
        result = paired_week_bootstrap(frame, scores, scores.copy(), k=1, n=100)
        for key in ["delta_precision_at_k", "delta_recall"]:
            self.assertEqual(result[key], {"estimate": 0, "ci95": [0, 0]})
        self.assertEqual(result["warnings_per_day"] if "warnings_per_day" in result else result["new"]["warnings_per_day"], 1)
        self.assertTrue(result["few_weeks"])
        self.assertTrue(result["limitations"])

    def test_issue_week_uses_d_plus_one_and_keeps_partial_boundary_blocks(self):
        one = population(days=7, start="2025-01-05")  # D Sunday -> issue Monday.
        result = paired_week_bootstrap(one, [0]*len(one), [0]*len(one), k=1, n=20)
        self.assertEqual(result["weeks"], 1)
        self.assertEqual(result["week_blocks"][0]["issue_week_start"], "2025-01-06")
        self.assertEqual(result["partial_week_blocks"], 0)
        self.assertIsNone(result["delta_precision_at_k"]["ci95"])
        partial = population(days=14, start="2025-01-01")  # Issue Thu..Wed: 4+7+3 days.
        result = paired_week_bootstrap(partial, [0]*len(partial), [0]*len(partial), k=1, n=20)
        self.assertEqual([week["days"] for week in result["week_blocks"]], [4, 7, 3])
        self.assertEqual(result["partial_week_blocks"], 2)
        self.assertEqual(result["min_days_per_block"], 3)
        self.assertEqual(result["max_days_per_block"], 7)
        self.assertEqual(result["rows"], 28)
        self.assertEqual(result["days"], 14)
        self.assertTrue(result["all_input_rows_included"])

    def test_block_ratios_use_totals_with_unequal_week_lengths(self):
        first = population(days=1, start="2025-01-05")
        second = population(days=7, start="2025-01-12")
        frame = pd.concat([first, second], ignore_index=True)
        new = np.r_[[1, 0], np.tile([0, 1], 7)]
        baseline = 1-new
        result = paired_week_bootstrap(frame, new, baseline, k=1, n=200, seed=84)
        self.assertEqual(result["new"]["precision_at_k"], 1/8)
        self.assertEqual(result["baseline"]["precision_at_k"], 7/8)
        self.assertEqual(result["delta_precision_at_k"]["estimate"], -.75)
        self.assertEqual(result["delta_recall"]["estimate"], -.75)
        self.assertEqual(result["delta_precision_at_k"]["ci95"], [-1, 1])
        self.assertEqual(result, paired_week_bootstrap(frame, new, baseline, k=1, n=200, seed=84))

    def test_zero_onset_resamples_are_explicitly_counted(self):
        frame = population(days=14)
        frame.loc[14:, ["onset_target", "alarm_target"]] = 0
        result = paired_week_bootstrap(frame, np.tile([1, 0], 14), np.tile([0, 1], 14), k=1, n=200)
        self.assertGreater(result["recall_replicates_without_onsets"], 0)
        self.assertLess(result["valid_recall_replicates"], 200)
        self.assertEqual(result["delta_recall"]["ci95"], [1, 1])


if __name__ == "__main__":
    unittest.main()
