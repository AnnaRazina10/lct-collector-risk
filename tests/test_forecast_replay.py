"""Input availability and daily correction checks, independent of raw archives."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/serving'))
import object_forecast as serving


def day_rows(day, alarm=0):
    row={'channel_id':'a','date':pd.Timestamp(day),**{c:0 for c in serving.features.COUNTS}}
    row.update(events_count=5,alarm_count=alarm,text_count=5)
    return pd.DataFrame([row])


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.seed=day_rows('2026-06-25')
        self.original=serving.DailyBatch('2026-06-26','2026-06-27T00:00:00+03:00',day_rows('2026-06-26',1))
        self.late=serving.DailyBatch('2026-06-26','2026-06-27T12:00:00+03:00',day_rows('2026-06-26',4))
        self.next=serving.DailyBatch('2026-06-27','2026-06-28T00:00:00+03:00',day_rows('2026-06-27'))

    def assemble(self,batches,date='2026-06-26'):
        return serving.assemble_history(self.seed,'2026-06-25',batches,date)

    def test_late_and_future_batches_do_not_change_previous_run(self):
        expected,first=self.assemble([self.original])
        result,second=self.assemble([self.original,self.late,self.next])
        pd.testing.assert_frame_equal(expected,result)
        self.assertEqual(first,second)
        revised,_=self.assemble([self.late,self.original,self.next],'2026-06-27')
        self.assertEqual(revised.loc[revised.date.eq('2026-06-26'),'alarm_count'].item(),4)

    def test_repeated_batches_are_idempotent_and_order_independent(self):
        a,p=self.assemble([self.original,self.next,self.late],'2026-06-27')
        b,q=self.assemble([self.late,self.original,self.next,self.original,self.next],'2026-06-27')
        pd.testing.assert_frame_equal(a,b);self.assertEqual(p,q)

    def test_ambiguous_versions_rejected_even_if_newer_seen_first(self):
        conflict=serving.DailyBatch(self.original.day,self.original.available_at,day_rows(self.original.day,3))
        with self.assertRaisesRegex(ValueError,'Conflicting'):
            self.assemble([self.late,self.original,conflict,self.next],'2026-06-27')

    def test_missing_package_is_not_silently_an_empty_day(self):
        with self.assertRaisesRegex(ValueError,'No complete'):
            self.assemble([self.late])
        with self.assertRaisesRegex(ValueError,'Missing complete'):
            self.assemble([self.next],'2026-06-27')
        empty=serving.DailyBatch(self.original.day,self.original.available_at,self.original.rows.iloc[:0])
        actual,_=self.assemble([empty])
        self.assertEqual(len(actual),len(self.seed))

    def test_malformed_count_and_incomplete_day_rejected(self):
        for value in [np.nan,np.inf,-1,.5,6]:
            bad=day_rows('2026-06-26');bad['alarm_count']=np.array([value],dtype=float)
            with self.subTest(value=value),self.assertRaises(ValueError):serving.validate_daily(bad)
        for at in ['2026-06-27','2026-06-26T23:59:59+03:00']:
            with self.subTest(at=at),self.assertRaises(ValueError):
                self.assemble([serving.DailyBatch(self.original.day,at,self.original.rows)])

    def test_feature_builder_does_not_read_or_create_future_observations(self):
        raw=pd.concat([day_rows('2025-01-01',1),day_rows('2025-01-02')])
        meta=pd.DataFrame({'channel_id':['a'],'object_id':['1']})
        objects=pd.DataFrame({'object_id':['1'],'object_kind':['x'],'parent_id':['0']})
        mart,_,_=serving.features.prepare_frames(raw,meta,objects,'2025-01-02')
        self.assertTrue(mart[serving.features.TARGET].isna().all())
        with self.assertRaisesRegex(ValueError,'future'):
            serving.features.prepare_frames(raw,meta,objects,'2025-01-01')
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            serving.features.prepare_frames(pd.concat([raw,raw]),meta,objects,'2025-01-02')

    def test_empty_archive_query_is_not_a_confirmed_daily_package(self):
        with patch.object(serving.pd,'read_parquet',return_value=day_rows('2026-06-30').iloc[:0]):
            with self.assertRaisesRegex(ValueError,'No confirmed archive package'):
                serving.read_daily('2026-07-01','2026-07-01')

    def test_model_cannot_be_replayed_before_its_selection_outcomes_exist(self):
        for day in ['2025-06-01','2025-11-29']:
            with self.subTest(day=day),self.assertRaisesRegex(ValueError,'unavailable'):
                serving.require_model_available(day)
        serving.require_model_available('2025-11-30')
        serving.require_model_available('2026-06-26')


if __name__=='__main__':unittest.main()
