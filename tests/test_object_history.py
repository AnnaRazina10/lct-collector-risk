import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/modeling'))
from object_history import COUNTS, TARGET, prepare_frames


def catalogs():
    meta = pd.DataFrame({'channel_id': ['a1', 'a2', 'b1', 'c1'],
                         'object_id': ['A', 'A', 'B', 'C']})
    objects = pd.DataFrame({'object_id': ['A', 'B', 'C'],
                            'object_kind': ['pump', None, 'future-kind'],
                            'parent_id': ['P', None, 'future-parent']})
    return meta, objects


def records(rows):
    frame = pd.DataFrame(rows, columns=['channel_id', 'date', 'events_count', 'alarm_count'])
    frame['date'] = pd.to_datetime(frame.date)
    for count in COUNTS:
        if count not in frame:
            frame[count] = 0
    return frame[['channel_id', 'date', *COUNTS]]


def build(raw, start='2023-01-01', end='2023-01-10', ids=('A', 'B')):
    return prepare_frames(raw, *catalogs(), start_date=start, end_date=end,
                          eligible_object_ids=ids)


class ObjectHistoryTests(unittest.TestCase):
    def test_complete_calendar_and_exact_d_plus_two_across_year_boundary(self):
        raw = records([('a1', '2022-12-30', 1, 0), ('a1', '2023-01-01', 1, 1)])
        mart, label_dates, _ = build(raw, '2022-12-30', '2023-01-03')
        a = mart[mart.object_id.eq('A')]
        self.assertEqual(len(mart), 10)
        self.assertEqual(a.date.tolist(), list(pd.date_range('2022-12-30', '2023-01-03')))
        self.assertEqual(a[TARGET].iloc[:3].tolist(), [1., 0., 0.])
        self.assertEqual(label_dates.loc[a.index].iloc[0], pd.Timestamp('2023-01-01'))
        self.assertTrue(a[TARGET].iloc[-2:].isna().all())
        self.assertTrue(label_dates.loc[a.index].iloc[-2:].isna().all())
        self.assertEqual(a.observed_today.tolist(), [1., 0., 1., 0., 0.])
        self.assertTrue(np.isnan(a.alarm_today_lag1.iloc[0]))
        self.assertEqual(a.alarm_today_lag1.iloc[3], 1.)
        self.assertEqual(a.target_dayofweek.iloc[0], 6)

    def test_feature_prefix_unchanged_by_later_rows_and_truncation(self):
        raw = records([('a1', '2023-01-01', 2, 1), ('a2', '2023-01-03', 1, 0),
                       ('b1', '2023-01-04', 1, 1), ('a1', '2023-01-08', 8, 7)])
        full, full_label_dates, _ = build(raw)
        cutoff = pd.Timestamp('2023-01-05')
        short, short_label_dates, _ = build(raw[raw.date.le(cutoff)], end=cutoff)
        feature_cols = [c for c in full if c != TARGET]
        prefix = full.loc[full.date.le(cutoff)]
        pd.testing.assert_frame_equal(prefix[feature_cols].reset_index(drop=True), short[feature_cols])
        known = short.date.le(cutoff-pd.Timedelta(days=2))
        pd.testing.assert_series_equal(prefix.loc[prefix.date.le(cutoff-pd.Timedelta(days=2)), TARGET].reset_index(drop=True), short.loc[known, TARGET].reset_index(drop=True))
        self.assertTrue(short.loc[~known, TARGET].isna().all())
        self.assertTrue(short_label_dates.loc[~known].isna().all())
        changed = raw.copy(); changed.loc[changed.date.gt(cutoff), COUNTS] = 1000000
        mutated, _, _ = build(changed)
        pd.testing.assert_frame_equal(full.loc[full.date.le(cutoff), feature_cols], mutated.loc[mutated.date.le(cutoff), feature_cols])
        self.assertTrue(full_label_dates[full.date.le(cutoff)].notna().all())

    def test_population_frozen_even_when_other_objects_appear_later(self):
        raw = records([('a1', '2023-01-01', 1, 0)])
        baseline, _, _ = build(raw, ids=('B', 'A'))
        extended = pd.concat([raw, records([('c1', '2023-01-03', 12, 8), ('unknown', '2023-01-02', 9, 2)])], ignore_index=True)
        result, _, coverage = build(extended, ids=('A', 'B'))
        pd.testing.assert_frame_equal(baseline, result)
        self.assertEqual(result.object_id.cat.categories.tolist(), ['A', 'B'])
        self.assertNotIn('future-kind', result.object_kind.cat.categories)
        self.assertEqual(coverage['excluded_noneligible_channel_days'], 1)
        self.assertEqual(coverage['excluded_unknown_channel_days'], 1)
        self.assertEqual(coverage['excluded_alarm_channel_days'], 1)
        self.assertEqual(coverage['included_channel_days'], 1)
        self.assertEqual(coverage['objects_with_observations_in_window'], 1)
        self.assertEqual(result.object_id.nunique(), 2)

    def test_group_history_and_catalog_denominators_are_preserved(self):
        raw = records([('a1', '2023-01-01', 2, 1), ('a2', '2023-01-01', 3, 0),
                       ('a1', '2023-01-02', 1, 1), ('b1', '2023-01-03', 1, 1)])
        mart, _, _ = build(raw, end='2023-01-04')
        a = mart[mart.object_id.eq('A')]
        b = mart[mart.object_id.eq('B')]
        self.assertEqual(a.events_count.iloc[0], 5)
        self.assertEqual(a.catalog_channels.iloc[0], 2)
        self.assertEqual(a.observed_fraction.iloc[0], 1)
        self.assertEqual(a.alarm_channel_fraction.iloc[0], .5)
        self.assertEqual(a.alarm_streak.tolist(), [1., 2., 0., 0.])
        self.assertEqual(a.days_since_alarm.tolist(), [0., 0., 1., 2.])
        np.testing.assert_allclose(a.historical_alarm_frequency, [1., 1., 2/3, .5])
        self.assertEqual(b.days_since_alarm.tolist(), [-1., -1., 0., 1.])
        self.assertTrue(np.isnan(b.alarm_today_lag1.iloc[0]))
        self.assertEqual(b.object_kind.iloc[0], 'unknown')
        self.assertEqual(len([c for c in mart if c not in ['date', TARGET]]), 71)

    def test_empty_records_keep_fixed_population_and_short_window_unknown_labels(self):
        mart, dates, _ = build(records([]), end='2023-01-02')
        self.assertEqual(len(mart), 4)
        self.assertTrue(mart.observed_today.eq(0).all())
        self.assertTrue(mart[TARGET].isna().all())
        self.assertTrue(dates.isna().all())
        self.assertTrue(mart.days_since_alarm.eq(-1).all())

    def test_explicit_calendar_origin_resets_history_without_hidden_prior(self):
        raw = records([('a1', '2023-01-05', 1, 1)])
        mart, _, _ = build(raw, start='2023-01-05', end='2023-01-07')
        a = mart[mart.object_id.eq('A')]
        self.assertEqual(a.historical_alarm_frequency.tolist(), [1., .5, 1/3])
        self.assertEqual(a.alarm_streak.tolist(), [1., 0., 0.])
        self.assertTrue(np.isnan(a.alarm_today_lag1.iloc[0]))

    def test_reject_future_prestart_unknown_and_non_midnight_dates(self):
        for date in ['2022-12-31', '2023-01-11', None, '2023-01-01 12:00:00']:
            with self.subTest(date=date), self.assertRaises(ValueError):
                build(records([('a1', date, 1, 1)]))
        with self.assertRaisesRegex(ValueError, 'start_date'):
            build(records([]), start='2023-01-11', end='2023-01-10')

    def test_reject_duplicate_records_metadata_and_unregistered_population(self):
        raw = records([('a1', '2023-01-01', 1, 0)])
        with self.assertRaisesRegex(ValueError, 'duplicate channel-day'):
            build(pd.concat([raw, raw], ignore_index=True))
        meta, objects = catalogs()
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            prepare_frames(raw, pd.concat([meta, meta.iloc[:1]]), objects,
                           start_date='2023-01-01', end_date='2023-01-10', eligible_object_ids=['A'])
        for ids in [[], ['A', 'A'], ['NOT-IN-CATALOG']]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                build(raw, ids=ids)


if __name__ == '__main__':
    unittest.main()
