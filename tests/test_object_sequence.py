import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/modeling'))
from object_sequence_candidate import (COUNT_COLUMNS, BINARY_COLUMNS, fit_scaler, daily_features,
                                       build_windows, complete_epoch_budget)


class SequenceWindowTests(unittest.TestCase):
    def fixture(self):
        frame = pd.MultiIndex.from_product([['A','B'],pd.date_range('2025-09-25',periods=8)],
                                           names=['object_id','date']).to_frame(index=False)
        frame['alarm_today'] = np.arange(len(frame))%2
        frame['observed_today'] = 1
        for c in COUNT_COLUMNS:
            frame[c] = np.arange(len(frame))+1
        return frame

    def test_window_stays_in_object_and_ends_today(self):
        frame = self.fixture()
        scaler = fit_scaler(frame)
        values = daily_features(frame,scaler)
        windows = build_windows(frame,scaler,4)
        np.testing.assert_array_equal(windows[:,-1],values)
        np.testing.assert_array_equal(windows[8,:-1],0)
        np.testing.assert_array_equal(windows[10,-3:],values[8:11])
        self.assertEqual(windows[8,-1,-1],1)

    def test_future_cannot_change_past_window_or_scaler(self):
        frame = self.fixture()
        scaler = fit_scaler(frame)
        before = build_windows(frame,scaler,4)
        changed = frame.copy()
        future = changed.date.gt('2025-09-28')
        changed.loc[future,COUNT_COLUMNS] = 100000
        changed.loc[future,BINARY_COLUMNS] = 0
        after = build_windows(changed,scaler,4)
        np.testing.assert_array_equal(before[~future],after[~future])
        self.assertEqual(scaler,fit_scaler(changed))

    def test_calendar_gap_is_rejected(self):
        frame = self.fixture().drop(index=2).reset_index(drop=True)
        with self.assertRaisesRegex(ValueError,'daily-contiguous'):
            build_windows(frame,fit_scaler(frame),4)

    def test_labels_are_not_features(self):
        frame = self.fixture()
        scaler = fit_scaler(frame)
        expected = build_windows(frame,scaler,4)
        frame['target_any_object_alarm_d_plus_2'] = 1
        frame['future_observed'] = 10000
        np.testing.assert_array_equal(expected,build_windows(frame,scaler,4))

    def test_refit_budget_rejects_incomplete_winning_epoch(self):
        with self.assertRaisesRegex(ValueError, 'complete winning epoch'):
            complete_epoch_budget({'best_epoch': 3, 'best_epoch_full_epoch': False},
                                  '/not-present/checkpoint.pt')
        self.assertEqual(complete_epoch_budget({'best_epoch': 36, 'best_epoch_full_epoch': True},
                                              '/not-present/checkpoint.pt'), 36)


if __name__ == '__main__':
    unittest.main()
