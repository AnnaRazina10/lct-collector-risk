import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/modeling'))
from sequence_features import carry_last, sequence_probabilities, recurrence, build_sequence_features


class SequenceTests(unittest.TestCase):
    def test_last_known_never_backfills_unknown(self):
        x, age=carry_last(np.array([[99,2,88,3]]),np.array([[False,True,False,True]]))
        np.testing.assert_equal(x,[[np.nan,2,2,3]])
        np.testing.assert_equal(age,[[np.nan,0,1,0]])

    def test_future_cannot_change_past_or_other_channel(self):
        rng=np.random.default_rng(4)
        obs=rng.integers(0,2,(3,40)); alarm=obs*rng.integers(0,2,(3,40))
        a2=alarm.copy(); o2=obs.copy();a2[0,20:]=1;o2[0,20:]=1
        for order in (1,2,3):
            a=sequence_probabilities(alarm,obs,order)
            b=sequence_probabilities(a2,o2,order)
            for x,y in zip(a,b):
                np.testing.assert_equal(x[:,:20],y[:,:20])
                np.testing.assert_equal(x[1:],y[1:])

    def test_pattern_forecast_learns_alternation(self):
        alarm=np.tile([0,1],100)[None,:];obs=np.ones_like(alarm)
        p,_,_=sequence_probabilities(alarm,obs,1)
        self.assertGreater(p[0,-2],.85)
        self.assertLess(p[0,-1],.15)

    def test_recurrence_uses_completed_gaps(self):
        age,gap,ema,cv=recurrence(np.array([[0,1,0,0,1,0,1]],bool))
        self.assertTrue(np.isnan(gap[0,3]))
        self.assertEqual(gap[0,4],3)
        self.assertEqual(age[0,5],1)
        self.assertEqual(gap[0,6],2)

    def test_full_feature_builder_causality_and_weekday_alignment(self):
        import pandas as pd
        rows=[]
        dates=pd.date_range('2025-01-01',periods=45)
        for channel in ['a','b']:
            for t,date in enumerate(dates):
                active=int(t%7==0); alarm=int(t%7==0)
                rows.append(dict(channel_id=channel,date=date,events_count=active,
                    alarm_count=alarm,fault_count=0,no_power_count=0,numeric_count=active,
                    value_mean_1d=10.,value_std_1d=1.,intra_last_event_is_alarm=alarm if active else np.nan,
                    intra_last_event_minute=800. if active else np.nan,
                    intra_last_alarm_minute=800. if alarm else np.nan))
        data=pd.DataFrame(rows);changed=data.copy()
        mask=(changed.channel_id=='a')&(changed.date>=dates[25])
        cols=[c for c in data if c not in ['channel_id','date']]
        changed.loc[mask,cols]=99
        a=build_sequence_features(data);b=build_sequence_features(changed)
        pd.testing.assert_frame_equal(a[~mask],b[~mask])
        # Day 6 predicts day 7; day 0 must contribute as the previous same weekday.
        self.assertAlmostEqual(a.seq_alarm_tomorrow_weekday_4w.iloc[6],1.1/11,places=6)
        self.assertAlmostEqual(a.seq_alarm_tomorrow_weekday_4w.iloc[7],.1/11,places=6)

if __name__=='__main__':unittest.main()
