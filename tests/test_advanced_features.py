import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/modeling'))
import numpy as np
from advanced_features import ewm, lag, streak

class DynamicsTests(unittest.TestCase):
    def test_lags_preserve_channel_boundary(self):
        np.testing.assert_equal(lag(np.array([[1,2,3],[7,8,9]]),1),[[0,1,2],[0,7,8]])
    def test_streak_resets_on_opposite_state(self):
        np.testing.assert_equal(streak(np.array([[1,1,0,1,0],[0,1,1,1,0]],dtype=bool)),[[1,2,0,1,0],[0,1,2,3,0]])
    def test_future_cannot_change_dynamics(self):
        a=np.array([[1,0,1,1,0]],dtype=bool);b=a.copy();b[:,3:]=~b[:,3:]
        for fn in [lambda x:ewm(x,7),lambda x:lag(x,2),streak]:
            np.testing.assert_equal(fn(a)[:,:3],fn(b)[:,:3])
    def test_ewm_recurrence(self):
        np.testing.assert_allclose(ewm(np.array([[0,1,0]],dtype=float),3),[[0,.5,.25]])

if __name__=='__main__':unittest.main()

class FullCausalityTest(unittest.TestCase):
    def test_future_perturbation_of_channel_and_neighbors(self):
        import pandas as pd
        from advanced_features import extend
        dates=pd.date_range('2025-01-01',periods=40)
        rows=[]
        for ch in ['a','b']:
            for i,d in enumerate(dates):
                rows.append(dict(channel_id=ch,date=d,object_id='unknown',sensor_type='test',
                    alarm_count=int(i%4==0),events_count=3,fault_count=0,no_power_count=0,
                    numeric_count=3,value_sum=30.,value_sumsq=300.))
        frame=pd.DataFrame(rows)
        changed=frame.copy();mask=changed.date>'2025-01-25'
        changed.loc[mask,['alarm_count','events_count','fault_count','value_sum']]=500
        a=extend(frame);b=extend(changed)
        pd.testing.assert_frame_equal(a[a.date<='2025-01-25'],b[b.date<='2025-01-25'])

class InferenceHelpersTest(unittest.TestCase):
    def test_predict_uses_supplied_feature_schema(self):
        import pandas as pd
        from experiment_v2 import predict
        class Dummy:
            def predict(self,x,num_threads):
                if list(x.columns)!=['signal']: raise AssertionError('Wrong schema')
                return x.signal.to_numpy()/10
        df=pd.DataFrame({'signal':[2.,4.],'unused':[9,9]})
        np.testing.assert_allclose(predict(Dummy(),'lgb',df,df.index,['signal'],[]),[.2,.4])
