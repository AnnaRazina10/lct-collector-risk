import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/modeling'))
from object_episode_audit import episode_labels, join_forecasts, summarize
from object_risk_probe import TARGET


def history(sequence, object_id='A', observed=None):
    return pd.DataFrame({'object_id': object_id,
                         'date': pd.date_range('2026-01-01', periods=len(sequence)),
                         'alarm_today': list(map(int, sequence)),
                         'observed_today': observed if observed is not None else 1})


class EpisodeAuditTests(unittest.TestCase):
    def test_registered_series_sequences(self):
        for sequence, expected in [('001110', [2]), ('101', [2]), ('1001', [3])]:
            result = episode_labels(history(sequence), 1)
            self.assertEqual(result.index[result.episode_start.eq(1)].tolist(), expected)
            self.assertTrue(np.isnan(result.episode_start.iloc[0]))
        self.assertEqual(episode_labels(history('1001'), 3).episode_start.iloc[-1], 0)
        self.assertEqual(episode_labels(history('10001'), 3).episode_start.iloc[-1], 1)

    def test_boundary_inclusion_for_each_gap(self):
        for gap in (1, 3, 7):
            sequence = '1' + '0' * (gap-1) + '1'
            self.assertEqual(episode_labels(history(sequence), gap).episode_start.iloc[-1], 0)
            outside = '1' + '0' * gap + '1'
            self.assertEqual(episode_labels(history(outside), gap).episode_start.iloc[-1], 1)

    def test_missing_calendar_and_initial_context_are_unknown(self):
        missing = history('00001').drop(index=2)
        result = episode_labels(missing, 3)
        self.assertFalse(result.context_known.iloc[-1])
        self.assertTrue(np.isnan(result.episode_start.iloc[-1]))
        self.assertEqual(result.gap_record_coverage.iloc[-1], 'unknown_context')
        result = episode_labels(history('001'), 3)
        self.assertTrue(result.episode_start.isna().all())
        # Explicit record-free calendar days remain eligible, with their own stratum.
        explicit = episode_labels(history('0001', observed=[0, 0, 0, 1]), 3)
        self.assertEqual(explicit.episode_start.iloc[-1], 1)
        self.assertEqual(explicit.gap_record_coverage.iloc[-1], 'no_records_in_gap')

    def test_objects_are_isolated_and_input_order_irrelevant(self):
        a = history('1111', 'A'); b = history('0001', 'B')
        mixed = pd.concat([a, b]).sample(frac=1, random_state=3)
        result = episode_labels(mixed, 3).set_index(['object_id', 'date'])
        self.assertEqual(result.loc[('A', pd.Timestamp('2026-01-04')), 'episode_start'], 0)
        self.assertEqual(result.loc[('B', pd.Timestamp('2026-01-04')), 'episode_start'], 1)

    def test_forecast_join_uses_exact_target_day(self):
        labels = episode_labels(history('00110'), 1)
        prediction = pd.DataFrame({'object_id': ['A', 'A'],
                                   'date': pd.to_datetime(['2026-01-03', '2026-01-01']),
                                   TARGET: [0, 1], 'selected': [.2, .8]})
        result = join_forecasts(prediction, labels.sample(frac=1, random_state=2))
        self.assertEqual(result.target_alarm.tolist(), [0, 1])
        self.assertEqual(result.episode_start.tolist(), [0, 1])
        self.assertTrue(((result.target_date-result.issue_date).dt.days == 1).all())
        self.assertTrue(((result.target_date-result.feature_date).dt.days == 2).all())
        with self.assertRaisesRegex(ValueError, 'exact D\\+2'):
            join_forecasts(prediction, labels[labels.date.ne('2026-01-03')])

    def test_warning_partition_preserves_repeat_alarm_success(self):
        labels = episode_labels(history('00110'), 1)
        prediction = pd.DataFrame({'object_id': ['A']*3,
                                   'date': pd.date_range('2026-01-01', periods=3),
                                   TARGET: [1, 1, 0], 'selected': [.8, .8, .8]})
        result = summarize(join_forecasts(prediction, labels), 'selected', .5)
        self.assertEqual(result['warnings_total'], 3)
        self.assertEqual(result['warned_episode_starts'], 1)
        self.assertEqual(result['warnings_on_repeated_alarm_days'], 1)
        self.assertEqual(result['warnings_on_no_alarm_days'], 1)
        self.assertEqual(result['any_alarm_true_warnings'], 2)
        self.assertEqual(result['warnings_without_new_episode_known_context'], 2)

    def test_gap_nesting_and_future_label_mutation(self):
        frame = history('100000000110010000000001')
        labels = [episode_labels(frame, gap) for gap in (1, 3, 7)]
        known = labels[-1].context_known
        self.assertTrue((labels[2].episode_start[known] <= labels[1].episode_start[known]).all())
        self.assertTrue((labels[1].episode_start[known] <= labels[0].episode_start[known]).all())
        changed = frame.copy(); changed.loc[changed.index > 12, 'alarm_today'] = 1
        for gap, before in zip((1, 3, 7), labels):
            after = episode_labels(changed, gap)
            pd.testing.assert_series_equal(before.episode_start.iloc[:13], after.episode_start.iloc[:13])

    def test_invalid_observation_or_duplicate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'requires'):
            episode_labels(history('01', observed=[0, 0]), 1)
        frame = history('01')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            episode_labels(pd.concat([frame, frame]), 1)

    def test_unknown_context_warning_is_not_an_assumed_negative(self):
        labels = episode_labels(history('1'), 1)
        prediction = pd.DataFrame({'object_id': ['A'], 'date': [pd.Timestamp('2025-12-30')],
                                   TARGET: [1], 'selected': [.8]})
        result = summarize(join_forecasts(prediction, labels), 'selected', .5)
        self.assertEqual(result['warnings_with_unknown_context'], 1)
        self.assertEqual(result['warnings_without_new_episode_known_context'], 0)
        self.assertEqual(result['original_any_alarm_precision'], 1)
        self.assertIsNone(result['episode_start_recall'])


if __name__ == '__main__':
    unittest.main()
