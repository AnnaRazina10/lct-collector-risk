"""Calendar and past-only transition feature invariants."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/modeling'))
from object_onset_experiment import onset_frame,EXTRA


class OnsetFeaturesTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame({'object_id':['a']*6+['b']*6,
            'date':list(pd.date_range('2025-01-01',periods=6))*2,
            'alarm_today':[1,0,1,1,0,1,0,0,0,0,0,0]})

    def test_exact_two_day_onset_and_unknown_future(self):
        frame=onset_frame(self.fixture())
        a=frame.loc[frame.object_id.eq('a')]
        np.testing.assert_array_equal(a.onset_target.iloc[:4],[1,0,0,1])
        self.assertTrue(a.onset_target.iloc[-2:].isna().all())
        self.assertTrue(frame.loc[frame.object_id.eq('b'),'onset_target'].iloc[:4].eq(0).all())

    def test_future_rows_do_not_change_transition_features(self):
        raw=self.fixture();full=onset_frame(raw)
        prefix=onset_frame(raw.loc[raw.date.le('2025-01-04')])
        pd.testing.assert_frame_equal(full.loc[full.date.le('2025-01-04'),EXTRA].reset_index(drop=True),prefix[EXTRA])
        raw.loc[raw.date.gt('2025-01-04'),'alarm_today']=1
        changed=onset_frame(raw)
        pd.testing.assert_frame_equal(full.loc[full.date.le('2025-01-04'),EXTRA],changed.loc[changed.date.le('2025-01-04'),EXTRA])

    def test_missing_calendar_and_duplicate_are_rejected(self):
        raw=self.fixture()
        with self.assertRaisesRegex(ValueError,'Incomplete'):onset_frame(raw.drop(index=2))
        with self.assertRaisesRegex(ValueError,'Duplicate'):onset_frame(pd.concat([raw,raw.iloc[:1]]))


if __name__=='__main__':unittest.main()
