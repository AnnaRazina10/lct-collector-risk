"""Sequence memory and observation-factorized forecasting; frozen temporal protocol."""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
import improve_baseline as v1
import experiment_v2 as v2
from sequence_features import build_sequence_features

ROOT=v1.ROOT
OUT=ROOT/'reports/improved_v3'
CACHE=ROOT/'data/interim/sequence_mart.parquet'
TARGET=v1.TARGET
OBS='label_observed_tomorrow'
CATS=['engineering_system','sensor_type','object_id','object_kind','channel_category']

def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def prepare():
    cols=['channel_id','date','alarm_count','events_count','fault_count','no_power_count',
          'numeric_count','value_mean_1d','value_std_1d']
    m=pd.read_parquet(v2.CACHE,columns=cols)
    intra=pd.concat([pd.read_parquet(ROOT/f'data/interim/intraday_{y}.parquet') for y in [2025,2026]])
    m=m.merge(intra,on=['channel_id','date'],how='left',sort=False,validate='one_to_one')
    features=build_sequence_features(m)
    features.to_parquet(CACHE,index=False)
    paths=[CACHE,ROOT/'src/modeling/sequence_features.py',ROOT/'src/modeling/experiment_v3.py',
           v2.OUT/'input_provenance.json']
    dump(OUT/'input_provenance.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})
    v1.log(f'Sequence cache ready: {features.shape}')

def load():
    for p,h in json.loads((OUT/'input_provenance.json').read_text()).items():
        if sha(ROOT/p)!=h:raise ValueError('Changed input '+p)
    m=v2.load()
    m=m.merge(pd.read_parquet(ROOT/'data/interim/channel_prior_frozen_v2.parquet'),on='channel_id',how='left',validate='many_to_one',sort=False)
    intra=pd.concat([pd.read_parquet(ROOT/f'data/interim/intraday_{y}.parquet') for y in [2025,2026]])
    m=m.merge(intra,on=['channel_id','date'],how='left',validate='one_to_one',sort=False)
    m=m.merge(pd.read_parquet(CACHE),on=['channel_id','date'],how='left',validate='one_to_one',sort=False)
    # Auxiliary LABEL only. Excluded explicitly from every model's features.
    nxt=m.groupby('channel_id',sort=False).events_count.shift(-1)
    m[OBS]=np.where(nxt.notna(),(nxt>0).astype(float),np.nan)
    return m

def sample(m,idx,label=TARGET,cap=1000000):
    y=m.loc[idx,label].to_numpy();p=idx[y==1];n=idx[y==0]
    rng=np.random.default_rng(139)
    pp=rng.choice(p,min(len(p),cap//3),replace=False)
    nn=rng.choice(n,min(len(n),cap-len(pp)),replace=False)
    ii=np.r_[pp,nn];rng.shuffle(ii)
    w=np.where(m.loc[ii,label]==1,len(p)/len(pp),len(n)/len(nn))
    return ii,w/w.mean()

def frame(m,idx,cfg):
    if TARGET in cfg['features'] or OBS in cfg['features']:raise ValueError('Label in features')
    x=v1.frame(m,idx,cfg['features'],cfg['cats'],cfg['kind']=='catboost')
    if cfg['kind']=='lgb':
        for col in cfg['cats']:x[col]=x[col].astype('category')
    return x

def component_predict(name,cfg,m,idx):
    x=frame(m,idx,cfg);artifact=cfg.get('artifact_name',name)
    if cfg['kind']=='catboost':
        model=CatBoostClassifier();model.load_model(str(OUT/'models'/f'{artifact}.cbm'))
        return model.predict_proba(x,thread_count=5)[:,1]
    model=lgb.Booster(model_file=str(OUT/'models'/f'{artifact}.txt'))
    return model.predict(x,num_threads=5)

def predict(name,configs,m,idx):
    c=configs[name]
    if c['kind']=='blend':return sum(w*predict(k,configs,m,idx) for k,w in c['parts'].items())
    if c['kind']=='product':return np.prod([predict(k,configs,m,idx) for k in c['parts']],axis=0)
    return component_predict(name,c,m,idx)

def fit(name,cfg,m,idx,valid_idx=None,refit=False):
    label=cfg.get('label',TARGET)
    if cfg.get('conditional_observed'):idx=idx[m.loc[idx,OBS].to_numpy()==1]
    ii,w=sample(m,idx,label)
    if cfg.get('half_life'):
        anchor=pd.Timestamp('2025-11-29' if refit else '2025-09-29')
        w*=np.power(.5,(anchor-m.loc[ii,'date']).dt.days.to_numpy()/cfg['half_life']);w/=w.mean()
    x=frame(m,ii,cfg);y=m.loc[ii,label].to_numpy()
    params=cfg['params'].copy()
    params['iterations' if cfg['kind']=='catboost' else 'n_estimators']=cfg['iterations'] if refit else 1100
    if not refit:
        vi=valid_idx
        if cfg.get('conditional_observed'):vi=vi[m.loc[vi,OBS].to_numpy()==1]
        xv=frame(m,vi,cfg);yv=m.loc[vi,label].to_numpy()
    start=time.time();v1.log(f'Training {name}: {len(ii)} rows, {len(cfg["features"])} features, refit={refit}')
    if cfg['kind']=='catboost':
        model=CatBoostClassifier(**params)
        model.fit(x,y,sample_weight=w,cat_features=cfg['cats'],
                  **({'eval_set':(xv,yv),'early_stopping_rounds':85,'verbose':150} if not refit else {'verbose':False}))
        iterations=model.tree_count_;suffix='.cbm'
    else:
        model=lgb.LGBMClassifier(**params)
        model.fit(x,y,sample_weight=w,categorical_feature=cfg['cats'],
                  **({'eval_set':[(xv,yv)],'callbacks':[lgb.early_stopping(85,first_metric_only=True),lgb.log_evaluation(150)]} if not refit else {}))
        model=model.booster_;iterations=model.current_iteration();suffix='.txt'
    artifact=('refit_' if refit else '')+name
    model.save_model(str(OUT/'models'/f'{artifact}{suffix}'))
    cfg.update(iterations=int(iterations),artifact_name=artifact)
    info={'rows':len(ii),'positives':int(y.sum()),'first_date':str(m.loc[ii,'date'].min()),
          'last_date':str(m.loc[ii,'date'].max()),'label':label,'conditional':bool(cfg.get('conditional_observed')),
          'seconds':time.time()-start,'iterations':iterations,'artifact_sha256':sha(OUT/'models'/f'{artifact}{suffix}')}
    if refit and iterations!=params['iterations' if cfg['kind']=='catboost' else 'n_estimators']:
        raise ValueError('Refit changed fixed tree count')
    dump(OUT/f'{artifact}_fit.json',info)
    del model,x;gc.collect()

def train():
    m=load();idx=m.index[m.date<='2025-09-29'];vi=m.index[(m.date>='2025-10-01')&(m.date<='2025-11-29')]
    features=[c for c in m if c not in ['channel_id','date',TARGET,OBS,'split','value_sum','value_sumsq','year','month']]
    configs={};scores={};rows=[]
    old=json.loads((v2.OUT/'selection_pre_calibration.json').read_text())['configs']
    for name in ['catboost_recent','lgb_interactions']:
        c=copy.deepcopy(old[name]);suffix='.cbm' if c['kind']=='catboost' else '.txt'
        new='v2_'+name;shutil.copy2(v2.OUT/'models'/f'{name}{suffix}',OUT/'models'/f'{new}{suffix}')
        c['v2_original']=name;configs[new]=c
    configs['v2_control']={'kind':'blend','parts':{'v2_catboost_recent':.5,'v2_lgb_interactions':.5}}
    scores['v2_control']=predict('v2_control',configs,m,vi)
    for name,kind,size,half,noid in [('lgb_memory31','lgb',31,None,False),('lgb_memory63','lgb',63,None,False),
        ('lgb_recent60','lgb',63,60,False),('cat_memory6','catboost',6,90,False),
        ('cat_memory8','catboost',8,90,False),('cat_no_identity','catboost',6,90,True),
        ('hurdle_activity','lgb',31,60,False),('hurdle_alarm','lgb',31,60,False)]:
        fs=[c for c in features if not(noid and c in ['channel_category','object_id'])]
        cats=[c for c in CATS if c in fs]
        params=dict(objective='binary',metric='average_precision',learning_rate=.035,num_leaves=size,
          min_child_samples=100,reg_lambda=12,colsample_bytree=.9,subsample=.9,subsample_freq=1,
          random_state=139,n_jobs=5,verbosity=-1) if kind=='lgb' else dict(depth=size,learning_rate=.05,
          loss_function='Logloss',eval_metric='PRAUC',l2_leaf_reg=10,random_seed=139,thread_count=5,allow_writing_files=False)
        cfg={'kind':kind,'features':fs,'cats':cats,'params':params,'half_life':half,
             'label':OBS if name=='hurdle_activity' else TARGET,'conditional_observed':name=='hurdle_alarm'}
        configs[name]=cfg;fit(name,cfg,m,idx,vi)
        scores[name]=predict(name,configs,m,vi)
        if not name.startswith('hurdle_'):rows.append(v2.record(name,m.loc[vi,TARGET],scores[name]))
        dump(OUT/'models/configs.json',configs)
        pd.DataFrame(rows).to_csv(OUT/'tuning_partial.csv',index=False)
    configs['hurdle']={'kind':'product','parts':['hurdle_activity','hurdle_alarm']}
    scores['hurdle']=scores.pop('hurdle_activity')*scores.pop('hurdle_alarm')
    rows.extend(v2.record(name,m.loc[vi,TARGET],scores[name]) for name in ['hurdle','v2_control'])
    pred=m.loc[vi,['channel_id','date',TARGET]].copy()
    for name,score in scores.items():pred[name]=score
    pred.to_parquet(OUT/'predictions/tune.parquet',index=False)
    pd.DataFrame(rows).to_csv(OUT/'tuning_candidates.csv',index=False);dump(OUT/'models/configs.json',configs)
    v1.log('Training complete; no new calibration or test evaluation.')

def select():
    pred=pd.read_parquet(OUT/'predictions/tune.parquet');y=pred[TARGET].to_numpy()
    configs=json.loads((OUT/'models/configs.json').read_text())
    names=[c for c in pred if c not in ['channel_id','date',TARGET]]
    rank=sorted(names,key=lambda n:average_precision_score(y,pred[n]),reverse=True)
    cat=max([n for n in names if n.startswith('cat_')],key=lambda n:average_precision_score(y,pred[n]))
    lg=max([n for n in names if n.startswith('lgb_')],key=lambda n:average_precision_score(y,pred[n]))
    pairs={'top2':rank[:2],'cat_lgb':[cat,lg],'hurdle':[next(n for n in rank if n!='hurdle'),'hurdle']}
    for label,pair in pairs.items():
        for w in [.25,.5,.75]:
            name=f'blend_{label}_{w}';parts={pair[0]:w,pair[1]:1-w}
            configs[name]={'kind':'blend','parts':parts};pred[name]=sum(weight*pred[n] for n,weight in parts.items())
    configs['blend_top3']={'kind':'blend','parts':{n:1/3 for n in rank[:3]}}
    pred['blend_top3']=pred[rank[:3]].mean(axis=1)
    rows=[v2.record(n,y,pred[n]) for n in pred if n not in ['channel_id','date',TARGET]]
    results=pd.DataFrame(rows).sort_values('average_precision',ascending=False)
    winner=results.iloc[0].model
    dump(OUT/'selection_pre_calibration.json',{'selected_model':winner,'configs':configs,
        'criterion':'Oct-Nov 2025 AP','manifest_sha256':sha(OUT/'input_provenance.json')})
    results.to_csv(OUT/'tuning_candidates.csv',index=False)
    v1.log('Selected '+winner+' before calibration and test');print(results[['model','average_precision']].to_string(index=False))

def finalize():
    cfg=json.loads((OUT/'selection_pre_calibration.json').read_text());m=load()
    if cfg['manifest_sha256']!=sha(OUT/'input_provenance.json'):raise ValueError('Manifest changed')
    configs=cfg['configs'];done=set();idx=m.index[m.date<='2025-11-29']
    def visit(name):
        if name in done:return
        c=configs[name]
        if c['kind'] in ['blend','product']:
            for n in c['parts']:visit(n)
        elif c.get('v2_original'):
            old=c['v2_original'];suffix='.cbm' if c['kind']=='catboost' else '.txt'
            artifact='refit_'+name
            shutil.copy2(v2.OUT/'models'/f'refit_{old}{suffix}',OUT/'models'/f'{artifact}{suffix}')
            c['artifact_name']=artifact
        else:fit(name,c,m,idx,refit=True)
        done.add(name)
    visit(cfg['selected_model']);dump(OUT/'selection_refit_before_calibration.json',cfg)
    vi=m.index[(m.date>='2025-12-01')&(m.date<='2025-12-30')]
    score=predict(cfg['selected_model'],configs,m,vi)
    t,op=v1.choose_threshold(m.loc[vi,TARGET],score);cfg.update(threshold=t,operating_points=op)
    dump(OUT/'selection.json',cfg);dump(OUT/'calibration_metrics.json',v1.metrics(m.loc[vi,TARGET],score,t))
    pred=m.loc[vi,['channel_id','date',TARGET]].copy();pred['score']=score;pred.to_parquet(OUT/'predictions/calibration.parquet',index=False)
    check=predict(cfg['selected_model'],configs,m,vi[:256]);np.testing.assert_allclose(check,score[:256],atol=1e-12)
    dump(OUT/'inference_check.json',{'rows':256,'maximum_difference':float(np.abs(check-score[:256]).max()),'passed':True})
    v1.log(f'Model and threshold frozen: {cfg["selected_model"]}, {t:.6f}')

def evaluate():
    cfg=json.loads((OUT/'selection.json').read_text());m=load()
    if cfg['manifest_sha256']!=sha(OUT/'input_provenance.json'):raise ValueError('Manifest changed')
    idx=v1.legacy.sample_eval(m[m.split=='test'],1200000,42).index
    pred=m.loc[idx,['channel_id','date',TARGET,'alarm_count','events_count']].copy()
    pred['score_v3']=predict(cfg['selected_model'],cfg['configs'],m,idx)
    old=pd.read_parquet(v2.OUT/'predictions/test.parquet')
    pred=pred.merge(old[['channel_id','date',TARGET,'score_v1','score_v2']],on=['channel_id','date',TARGET],validate='one_to_one')
    if len(pred)!=1200000:raise ValueError('Different test rows')
    t2=json.loads((v2.OUT/'selection.json').read_text())['threshold']
    rows=[{'model':'v2',**v1.metrics(pred[TARGET],pred.score_v2,t2)},
          {'model':'v3',**v1.metrics(pred[TARGET],pred.score_v3,cfg['threshold'])}]
    pd.DataFrame(rows).to_csv(OUT/'test_comparison.csv',index=False)
    pred.to_parquet(OUT/'predictions/test.parquet',index=False)
    monthly=[]
    for month,p in pred.groupby(pred.date.dt.to_period('M')):
        monthly.append({'month':str(month),'rows':len(p),'v2_AP':average_precision_score(p[TARGET],p.score_v2),'v3_AP':average_precision_score(p[TARGET],p.score_v3)})
    pd.DataFrame(monthly).to_csv(OUT/'test_monthly.csv',index=False)
    dump(OUT/'test_done.json',{'selection_sha256':sha(OUT/'selection.json'),'metrics':rows})
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','train','select','finalize','evaluate']);args=p.parse_args()
    for d in [OUT,OUT/'models',OUT/'predictions']:d.mkdir(parents=True,exist_ok=True)
    globals()[args.phase]()
