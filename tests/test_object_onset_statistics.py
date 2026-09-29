import io
import sys
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/modeling'))
from object_onset_statistics import (NUMERIC_INPUTS, OnsetLogistic,
                                     SmoothedOnsetFrequency, causal_design)


def fixture():
    rng = np.random.default_rng(12)
    frame = pd.DataFrame({'object_id': ['A']*30+['B']*30,
                          'date': list(pd.date_range('2025-01-01', periods=30))*2})
    for name in NUMERIC_INPUTS:
        frame[name] = rng.integers(0, 5, len(frame)).astype(float)
    frame['alarm_today'] = rng.integers(0, 2, len(frame))
    frame['observed_today'] = 1
    frame['onset_target'] = np.arange(len(frame))%3 == 0
    frame.loc[0, 'days_since_alarm'] = -1
    frame.loc[2, 'alarm_channel_fraction'] = np.nan
    return frame


class OnsetStatisticsTests(unittest.TestCase):
    def test_design_is_prefix_and_future_mutation_invariant(self):
        frame = fixture(); cutoff = pd.Timestamp('2025-01-14')
        past = frame.date.le(cutoff)
        expected = causal_design(frame.loc[past])
        changed = frame.copy()
        changed.loc[~past, list(NUMERIC_INPUTS)] = 1000000
        changed['future_observation'] = 999
        changed['onset_target'] = 'unreadable future labels'
        pd.testing.assert_frame_equal(expected, causal_design(changed).loc[past])
        self.assertNotIn('onset_target', expected)
        self.assertEqual(expected.never_registered_alarm.iloc[0], 1)
        self.assertEqual(expected.days_since_alarm.iloc[0], 0)

    def test_calendar_is_known_d_plus_2(self):
        frame = fixture().iloc[:1]
        design = causal_design(frame)
        # 2025-01-01 is Wednesday, D+2 is Friday (weekday 4).
        self.assertAlmostEqual(design.target_week_sin.iloc[0], np.sin(2*np.pi*4/7))

    def test_frozen_preprocessing_unknown_object_and_reload(self):
        train = fixture()
        estimator = OnsetLogistic().fit(train)
        unknown = train.iloc[:5].drop(columns='onset_target').copy()
        unknown['object_id'] = 'UNSEEN'
        unknown['object_kind'] = 'UNSEEN_KIND'
        unknown['parent_id'] = 'UNSEEN_PARENT'
        scaler = estimator.pipeline_.named_steps['preprocess'].named_transformers_['numeric'].named_steps['scaler']
        means = scaler.mean_.copy()
        expected = estimator.predict(unknown)
        self.assertTrue(np.isfinite(expected).all())
        self.assertTrue(((expected > 0) & (expected < 1)).all())
        altered = unknown.copy(); altered['onset_target'] = 'must not be read'; altered['future_alarm'] = 100
        np.testing.assert_array_equal(expected, estimator.predict(altered))
        np.testing.assert_array_equal(scaler.mean_, means)
        buffer = io.BytesIO(); joblib.dump(estimator, buffer); buffer.seek(0)
        np.testing.assert_array_equal(expected, joblib.load(buffer).predict(unknown))
        self.assertEqual(estimator.training_rows_, len(train))

    def test_predictions_independent_of_other_prediction_rows(self):
        frame = fixture(); estimator = OnsetLogistic().fit(frame)
        before = estimator.predict(frame.iloc[:3])
        changed = frame.copy(); changed.loc[3:, list(NUMERIC_INPUTS)] = 1000000
        np.testing.assert_allclose(before, estimator.predict(changed)[:3], rtol=0, atol=1e-12)

    def test_frequency_formula_unknown_and_target_ignored(self):
        train = pd.DataFrame({'object_id': ['A', 'A', 'B', 'B'], 'onset_target': [1, 1, 0, 0]})
        estimator = SmoothedOnsetFrequency().fit(train)
        query = pd.DataFrame({'object_id': ['A', 'B', 'UNKNOWN'], 'onset_target': ['bad']*3})
        expected = np.array([17/32, 15/32, .5])
        np.testing.assert_allclose(estimator.predict(query), expected, rtol=0, atol=0)
        buffer = io.BytesIO(); joblib.dump(estimator, buffer); buffer.seek(0)
        np.testing.assert_array_equal(estimator.predict(query), joblib.load(buffer).predict(query))
        self.assertEqual(estimator.training_rows_, 4)

    def test_fit_rejects_unknown_labels_and_predict_rejects_missing_causal_inputs(self):
        frame = fixture(); bad = frame.copy(); bad['onset_target'] = np.nan
        for estimator in [OnsetLogistic(), SmoothedOnsetFrequency()]:
            with self.assertRaisesRegex(ValueError, 'known'):
                estimator.fit(bad)
        with self.assertRaisesRegex(ValueError, 'Missing causal'):
            causal_design(frame.drop(columns='observed_today'))


if __name__ == '__main__':
    unittest.main()
