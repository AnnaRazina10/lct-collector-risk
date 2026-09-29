"""Predeclared monthly refitting with completed past labels only."""
import copy
import gc
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import improve_baseline as v1
import experiment_v2 as v2
from refit_components import _lgb_params, _sha
from feature_rows import V2RowReader

ROOT=v1.ROOT
OUT=ROOT/'reports/rolling_update'

def main():
    for d in [OUT,OUT/'models',OUT/'predictions']:d.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((v2.OUT/'selection.json').read_text())
    for path,expected in json.loads((v2.OUT/'input_provenance.json').read_text()).items():
        if _sha(ROOT/path)!=expected:raise ValueError('Input changed: '+path)
    m=pd.read_parquet(v2.CACHE,columns=['channel_id','date',v1.TARGET,'split'])
    component=cfg['configs']['lgb_interactions'];catcfg=cfg['configs']['catboost_recent']
    reader=V2RowReader(ROOT,list(dict.fromkeys(component['features']+catcfg['features'])))
    source=v2.model_load('lgb_interactions',component);cat=v2.model_load('catboost_recent',catcfg)
    params=_lgb_params(source);params['num_threads']=3
    trees=source.current_iteration()
    ci=m.index[(m.date>='2025-12-01')&(m.date<='2025-12-30')]
    calibration=reader.read(ci)
    lcal=v2.predict(source,'lgb',calibration,ci,component['features'],component['cats'])
    ccal=v2.predict(cat,'catboost',calibration,ci,catcfg['features'],catcfg['cats'])
    thresholds={name:v1.choose_threshold(m.loc[ci,v1.TARGET],s)[0] for name,s in
                [('rolling_lgb',lcal),('rolling_blend',.5*lcal+.5*ccal)]}
    previous=pd.read_parquet(v2.OUT/'predictions/calibration.parquet')
    checked=calibration[['channel_id','date']].assign(score=.5*lcal+.5*ccal).merge(previous,on=['channel_id','date'],validate='one_to_one')
    maximum_difference=float(np.abs(checked.score_x-checked.score_y).max())
    if len(checked)!=len(ci) or maximum_difference>1e-12:raise ValueError('Sparse reader changed calibration scores')
    v2.dump(OUT/'reader_check.json',{'rows':len(checked),'maximum_difference':maximum_difference,'passed':True})
    del calibration,previous,checked;gc.collect()
    v2.dump(OUT/'frozen_before_test.json',{'trees':trees,'parameters':params,'thresholds':thresholds,
        'blend_weights':{'updated_lgb':.5,'frozen_catboost':.5},'feature_list':component['features'],
        'source_selection_sha256':_sha(v2.OUT/'selection.json'),'protocol_sha256':_sha(OUT/'protocol.md')})
    idx=v1.legacy.sample_eval(m[m.split=='test'],1200000,42).index
    pred=m.loc[idx,['channel_id','date',v1.TARGET]].copy()
    pred['rolling_lgb']=np.nan;pred['rolling_blend']=np.nan
    checks=[]
    for month in sorted(pred.date.dt.to_period('M').unique()):
        start=month.start_time;cutoff=start-pd.Timedelta(days=2)
        ti=m.index[m.date<=cutoff];si,w=v2.sample(m,ti)
        assert (m.loc[si,'date'].max()+pd.Timedelta(days=1)) < start
        ii=pred.index[pred.date.dt.to_period('M')==month]
        local=reader.read(pd.Index(np.union1d(si,ii)))
        x=v1.frame(local,si,component['features'],component['cats'])
        for col in component['cats']:x[col]=x[col].cat.remove_unused_categories()
        data=lgb.Dataset(x,label=m.loc[si,v1.TARGET],weight=w,categorical_feature=component['cats'])
        v1.log(f'Monthly update {month}: rows={len(si)}, training through={cutoff.date()}')
        t=time.time();model=lgb.train(params,data,num_boost_round=trees)
        model.save_model(str(OUT/'models'/f'lgb_{month}.txt'))
        lp=model.predict(v1.frame(local,ii,component['features'],component['cats']),num_threads=3)
        cp=v2.predict(cat,'catboost',local,ii,catcfg['features'],catcfg['cats'])
        pred.loc[ii,'rolling_lgb']=lp;pred.loc[ii,'rolling_blend']=.5*lp+.5*cp
        checks.append({'month':str(month),'training_feature_end':str(m.loc[si,'date'].max().date()),
           'training_label_end':str((m.loc[si,'date'].max()+pd.Timedelta(days=1)).date()),
           'first_test_feature_date':str(m.loc[ii,'date'].min().date()),'rows':len(ii),
           'training_rows':len(si),'trees':model.current_iteration(),'seconds':time.time()-t})
        v2.dump(OUT/'temporal_checks.json',checks)
        del x,data,model,local;gc.collect()
    if pred[['rolling_lgb','rolling_blend']].isna().any().any():raise ValueError('Missing monthly scores')
    old=pd.read_parquet(v2.OUT/'predictions/test.parquet')
    pred=pred.merge(old[['channel_id','date',v1.TARGET,'score_v2']],on=['channel_id','date',v1.TARGET],validate='one_to_one')
    if len(pred)!=1200000:raise ValueError('Test identities differ')
    rows=[{'model':'frozen_v2',**v1.metrics(pred[v1.TARGET],pred.score_v2,cfg['threshold'])}]
    for name in thresholds:rows.append({'model':name,**v1.metrics(pred[v1.TARGET],pred[name],thresholds[name])})
    pd.DataFrame(rows).to_csv(OUT/'comparison.csv',index=False)
    monthly=[]
    for month,p in pred.groupby(pred.date.dt.to_period('M')):
        for name,col,threshold in [('frozen_v2','score_v2',cfg['threshold'])]+[(n,n,t) for n,t in thresholds.items()]:
            monthly.append({'month':str(month),'model':name,**v1.metrics(p[v1.TARGET],p[col],threshold)})
    pd.DataFrame(monthly).to_csv(OUT/'monthly.csv',index=False)
    pred.to_parquet(OUT/'predictions/test.parquet',index=False)
    v2.dump(OUT/'completed.json',{'rows':len(pred),'months':len(checks),'frozen_config_sha256':_sha(OUT/'frozen_before_test.json'),
        'code_sha256':_sha(Path(__file__)),'interpretation':'Prequential monthly update, not independent static holdout'})
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':main()
