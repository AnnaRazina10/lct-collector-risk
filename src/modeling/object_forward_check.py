"""Evaluate the previously frozen +24..48h object model, never select on 2026."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import object_risk_probe as source

ROOT=source.ROOT
OUT=ROOT/'reports/object_forward_check'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((source.OUT/'configuration.json').read_text())
    model_path=ROOT/cfg['model_path']
    if source.checksum(model_path)!=cfg['model_sha256']:raise ValueError('Original model changed')
    source.dump(OUT/'frozen_before_test.json',{'original_configuration':cfg,
        'policy':'Original model and Oct-Nov thresholds, no refit or test selection',
        'feature_dates':['2025-12-31','2026-06-28'],'issue_dates':['2026-01-01','2026-06-29'],
        'target_dates':['2026-01-02','2026-06-30'],'unit':'object-day','minimum_lead_hours':24})
    m,label_date,coverage=source.prepare('2026-06-30')
    model=lgb.Booster(model_file=str(model_path))
    tune=m[m.date.between('2025-10-01','2025-11-28')].copy()
    score=model.predict(tune[cfg['features']],num_threads=2)
    old=pd.read_parquet(source.CACHE/'object_risk_tuning_predictions.parquet')
    joined=tune[['object_id','date',source.TARGET]].assign(reproduced=score).merge(old,on=['object_id','date',source.TARGET],validate='one_to_one')
    if len(joined)!=4602:raise ValueError('Original validation rows changed')
    difference=float(np.max(np.abs(joined.reproduced-joined.lightgbm_object)))
    if difference>1e-12:raise ValueError('Extended features altered historical predictions')
    part=m[m.date.between('2025-12-31','2026-06-28')].copy()
    y=part[source.TARGET].to_numpy(dtype=int);days=part.date.nunique()
    if not label_date.loc[part.index].eq(part.date+pd.Timedelta(days=2)).all():raise ValueError('Target time mismatch')
    frequency=m[m.date<='2025-09-28'].groupby('object_id',observed=True)[source.TARGET].mean()
    scores={'always_positive':np.ones(len(part)),'persistence_alarm_today':part.alarm_today.to_numpy(),
        'frozen_object_frequency':part.object_id.map(frequency).astype(float).to_numpy(),
        'lightgbm_object':model.predict(part[cfg['features']],num_threads=2)}
    rows=[]
    for name,s in scores.items():
        for cohort,mask in [('all_object_days',np.ones(len(part),bool)),('no_alarm_on_feature_day',part.alarm_today.eq(0).to_numpy())]:
            rows.append(source.metrics(name,cohort,y[mask],s[mask],cfg['thresholds'][name],days))
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    source.dump(OUT/'checks.json',{'historical_rows_reproduced':len(joined),'maximum_difference':difference,
        'test_rows':len(part),'days':int(days),'objects':int(part.object_id.nunique()),'coverage':coverage,
        'model_sha256':source.checksum(model_path),'code_sha256':source.checksum(Path(__file__)),
        'feature_code_sha256':source.checksum(Path(source.__file__)),'passed':True})
    pred=part[['object_id','date',source.TARGET]].copy()
    for name,s in scores.items():pred[name]=s
    pred.to_parquet(source.CACHE/'object_forward_predictions.parquet',index=False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
