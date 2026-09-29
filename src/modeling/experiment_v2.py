#!/usr/bin/env python3
"""Predeclared causal features, GBDT diversity, temporal blending, held-back test."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve
import improve_baseline as v1
from advanced_features import extend

ROOT=v1.ROOT
OUT=ROOT/'reports/improved_v2'
CACHE=ROOT/'data/interim/advanced_mart.parquet'
TARGET=v1.TARGET

def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str))

def prepare():
    start=time.time()
    mart=v1.augment(v1.prepare())
    mart=extend(mart)
    mart.to_parquet(CACHE,index=False)
    dump(OUT/'feature_cache.json',{'rows':len(mart),'columns':len(mart.columns),'seconds':time.time()-start,
         'source_fingerprint':v1.fingerprint(),'feature_sha256':hashlib.sha256((ROOT/'src/modeling/advanced_features.py').read_bytes()).hexdigest()})
    v1.log(f'Advanced feature cache: {mart.shape}, {time.time()-start:.1f}s')

def freeze():
    import shutil
    shutil.copy2(ROOT/'data/interim/channel_prior_pre2025.parquet',ROOT/'data/interim/channel_prior_frozen_v2.parquet')
    shutil.copy2(ROOT/'data/interim/channel_prior_pre2025.metadata.json',OUT/'history_provenance.json')
    relatives=['data/interim/advanced_mart.parquet','data/interim/channel_prior_frozen_v2.parquet',
               'data/interim/intraday_2025.parquet','data/interim/intraday_2026.parquet']
    manifest={}
    for relative in relatives:
        digest=hashlib.sha256()
        with (ROOT/relative).open('rb') as source:
            for chunk in iter(lambda:source.read(1<<20),b''):digest.update(chunk)
        manifest[relative]=digest.hexdigest()
    dump(OUT/'input_provenance.json',manifest)
    v1.log('Inputs frozen. Train all candidates again after replacing an earlier freeze.')


def load():
    manifest=OUT/'input_provenance.json'
    if manifest.exists():
        for relative,expected in json.loads(manifest.read_text()).items():
            digest=hashlib.sha256()
            with (ROOT/relative).open('rb') as source:
                for chunk in iter(lambda:source.read(1<<20),b''):digest.update(chunk)
            if digest.hexdigest()!=expected:raise ValueError('Input changed since training: '+relative)
    return pd.read_parquet(CACHE)

def sample(mart, idx, cap=1000000):
    y=mart.loc[idx,TARGET].to_numpy()
    p=idx[y==1]; n=idx[y==0]; rng=np.random.default_rng(84)
    pp=rng.choice(p,min(len(p),cap//3),replace=False)
    nn=rng.choice(n,min(len(n),cap-len(pp)),replace=False)
    out=np.r_[pp,nn]; rng.shuffle(out)
    yy=mart.loc[out,TARGET].to_numpy()
    w=np.where(yy==1,len(p)/len(pp),len(n)/len(nn))
    return out,w/w.mean()

def predict(model,kind,mart,idx,features,cats):
    x=v1.frame(mart,idx,features,cats,kind=='catboost')
    if kind=='catboost': return model.predict_proba(x,thread_count=5)[:,1]
    return model.predict(x,num_threads=5)

def model_load(name, cfg):
    name=cfg.get('artifact_name',name)
    if cfg['kind']=='catboost':
        m=CatBoostClassifier(); m.load_model(str(OUT/'models'/f'{name}.cbm'));return m
    return lgb.Booster(model_file=str(OUT/'models'/f'{name}.txt'))

def record(name,y,s,extra=None):
    t,op=v1.choose_threshold(y,s)
    row={'model':name,**v1.metrics(y,s,t),**op,**(extra or {})}
    return row

def train(args):
    mart=load()
    if args.prior:
        prior=pd.read_parquet(args.prior)
        mart=mart.merge(prior,on='channel_id',how='left',validate='many_to_one',sort=False)
    if args.intraday:
        intra=pd.concat([pd.read_parquet(ROOT/f'data/interim/intraday_{y}.parquet') for y in [2025,2026]],ignore_index=True)
        mart=mart.merge(intra,on=['channel_id','date'],how='left',validate='one_to_one',sort=False)
    # No target or future-derived columns are eligible.
    features=[c for c in mart if c not in ['date','channel_id',TARGET,'split','value_sum','value_sumsq','year','month']]
    cats=['engineering_system','sensor_type','object_id','object_kind','channel_category']
    for c in features:
        if c not in cats and not pd.api.types.is_numeric_dtype(mart[c]):raise ValueError('Non-numeric feature '+c)
    train_idx=mart.index[mart.date<='2025-09-29']
    tune_idx=mart.index[(mart.date>='2025-10-01')&(mart.date<='2025-11-29')]
    cal_idx=mart.index[(mart.date>='2025-12-01')&(mart.date<='2025-12-30')]
    idx,w=sample(mart,train_idx)
    yt=mart.loc[idx,TARGET].to_numpy(); yv=mart.loc[tune_idx,TARGET].to_numpy()
    scores={}; configs={}; results=[]
    oldcfg=json.loads((ROOT/'reports/improved_v1/selection.json').read_text())
    old=CatBoostClassifier();old.load_model(str(ROOT/'reports/improved_v1/models/catboost_history.cbm'))
    scores['v1_control']=predict(old,'catboost',mart,tune_idx,oldcfg['features'],oldcfg['categorical_features'])
    configs['v1_control']={'kind':'catboost','features':oldcfg['features'],'cats':oldcfg['categorical_features'],'external_v1':True}
    results.append(record('v1_control',yv,scores['v1_control']))
    specs=[('lgb_dynamics','lgb',31,False),('lgb_interactions','lgb',63,False),
           ('lgb_no_intraday','lgb',31,False),('lgb_no_old_history','lgb',31,False),
           ('catboost_dynamics','catboost',8,False),('catboost_recent','catboost',6,True)]
    for name,kind,size,recent in specs:
        start=time.time()
        selected_features=[f for f in features if not (name=='lgb_no_intraday' and f.startswith('intra_')) and not (name=='lgb_no_old_history' and f.startswith('old_'))]
        v1.log(f'Training {name}, {len(selected_features)} features')
        x=v1.frame(mart,idx,selected_features,cats,kind=='catboost')
        xv=v1.frame(mart,tune_idx,selected_features,cats,kind=='catboost')
        ww=w.copy()
        if recent:
            age=(pd.Timestamp('2025-09-29')-mart.loc[idx,'date']).dt.days.to_numpy()
            ww*=np.power(.5,age/90);ww/=ww.mean()
        if kind=='lgb':
            m=lgb.LGBMClassifier(objective='binary',metric='average_precision',
                n_estimators=1000,learning_rate=.035,num_leaves=size,min_child_samples=120,
                reg_lambda=10,colsample_bytree=.85,subsample=.85,subsample_freq=1,
                random_state=84,n_jobs=5,verbosity=-1)
            m.fit(x,yt,sample_weight=ww,eval_set=[(xv,yv)],categorical_feature=cats,
                  callbacks=[lgb.early_stopping(70,first_metric_only=True),lgb.log_evaluation(100)])
            m=m.booster_;s=m.predict(xv,num_threads=5);iterations=m.current_iteration()
            m.save_model(str(OUT/'models'/f'{name}.txt'))
        else:
            m=CatBoostClassifier(iterations=900,depth=size,learning_rate=.055,
                loss_function='Logloss',eval_metric='PRAUC',l2_leaf_reg=8,
                random_seed=84,thread_count=5,allow_writing_files=False)
            m.fit(x,yt,sample_weight=ww,cat_features=cats,eval_set=(xv,yv),early_stopping_rounds=70,verbose=100)
            s=m.predict_proba(xv,thread_count=5)[:,1];iterations=m.tree_count_
            m.save_model(str(OUT/'models'/f'{name}.cbm'))
        scores[name]=s;configs[name]={'kind':kind,'features':selected_features,'cats':cats,'iterations':iterations}
        results.append(record(name,yv,s,{'seconds':time.time()-start,'iterations':iterations}))
        pd.DataFrame(results).to_csv(OUT/'tuning_candidates.csv',index=False)
        dump(OUT/'models/configs.json',configs)
        del x,xv,m;gc.collect()
    # System experts have shared target and same tune rows. Train-only eligibility.
    v1.log('Training system specialists')
    specialist_scores=scores['lgb_dynamics'].copy();specialists={}
    start=time.time()
    for system in mart.loc[idx,'engineering_system'].unique():
        si=idx[mart.loc[idx,'engineering_system'].to_numpy()==system]
        vi=tune_idx[mart.loc[tune_idx,'engineering_system'].to_numpy()==system]
        yy=mart.loc[si,TARGET].to_numpy();vy=mart.loc[vi,TARGET].to_numpy()
        if yy.sum()<300 or len(vi)==0 or len(np.unique(vy))<2:continue
        name='expert_'+str(len(specialists));xf=v1.frame(mart,si,features,cats);vf=v1.frame(mart,vi,features,cats)
        m=lgb.LGBMClassifier(objective='binary',metric='average_precision',n_estimators=600,
            learning_rate=.04,num_leaves=15,min_child_samples=80,reg_lambda=15,
            colsample_bytree=.85,random_state=84,n_jobs=5,verbosity=-1)
        subw=pd.Series(w,index=idx).loc[si].to_numpy()
        m.fit(xf,yy,sample_weight=subw,eval_set=[(vf,vy)],categorical_feature=cats,
              callbacks=[lgb.early_stopping(50,first_metric_only=True)])
        specialist_scores[np.isin(tune_idx,vi)]=m.predict_proba(vf)[:,1]
        m.booster_.save_model(str(OUT/'models'/f'{name}.txt'));specialists[str(system)]=name
        del xf,vf,m;gc.collect()
    scores['system_experts']=specialist_scores
    configs['system_experts']={'kind':'experts','features':features,'cats':cats,'specialists':specialists,'fallback':'lgb_dynamics'}
    results.append(record('system_experts',yv,specialist_scores,{'seconds':time.time()-start}))
    dump(OUT/'models/configs.json',configs)
    pd.DataFrame(results).to_csv(OUT/'tuning_candidates.csv',index=False)
    predictions=mart.loc[tune_idx,['channel_id','date',TARGET]].copy()
    for name,score in scores.items(): predictions[name]=score
    predictions.to_parquet(OUT/'predictions/tune_base.parquet',index=False)
    dump(OUT/'training_context.json',{'prior_path':str(args.prior) if args.prior else None,
         'intraday':args.intraday,'train_rows':len(idx),'features':features})
    v1.log('Training complete; ensemble/threshold selection deferred; no test evaluated')


def select(choose_only=False):
    context=json.loads((OUT/'training_context.json').read_text())
    mart=load()
    if context['prior_path']:
        mart=mart.merge(pd.read_parquet(context['prior_path']),on='channel_id',how='left',validate='many_to_one',sort=False)
    if context['intraday']:
        intra=pd.concat([pd.read_parquet(ROOT/f'data/interim/intraday_{y}.parquet') for y in [2025,2026]],ignore_index=True)
        mart=mart.merge(intra,on=['channel_id','date'],how='left',validate='one_to_one',sort=False)
    tune_idx=mart.index[(mart.date>='2025-10-01')&(mart.date<='2025-11-29')]
    cal_idx=mart.index[(mart.date>='2025-12-01')&(mart.date<='2025-12-30')]
    preds=pd.read_parquet(OUT/'predictions/tune_base.parquet')
    keys=['channel_id','date',TARGET]
    if not np.array_equal(preds[keys].astype(str).to_numpy(),mart.loc[tune_idx,keys].astype(str).to_numpy()):
        raise ValueError('Tuning prediction row identities do not match feature matrix')
    yv=preds[TARGET].to_numpy()
    scores={name:preds[name].to_numpy() for name in preds if name not in ['channel_id','date',TARGET]}
    configs=json.loads((OUT/'models/configs.json').read_text())
    results=pd.read_csv(OUT/'tuning_candidates.csv').to_dict('records')
    tabm_path=OUT/'predictions/tabm_tune.parquet'
    if tabm_path.exists():
        neural=pd.read_parquet(tabm_path)
        joined=preds[keys].merge(neural[keys+['score']],on=keys,how='left',validate='one_to_one')
        if len(neural)!=len(preds) or joined.score.isna().any():raise ValueError('TabM tuning identities mismatch')
        scores['tabm']=joined.score.to_numpy()
        configs['tabm']={'kind':'tabm'}
        results.append(record('tabm',yv,scores['tabm']))
    # v1 previously used December for early stopping; comparator only.
    eligible=[name for name in scores if name!='v1_control']
    ranking=sorted(eligible,key=lambda name:average_precision_score(yv,scores[name]),reverse=True)
    blends={}
    for weight in [.25,.5,.75]:
        name=f'blend_top2_{weight}'
        blends[name]={ranking[0]:weight,ranking[1]:1-weight}
    blends['blend_top3']={name:1/3 for name in ranking[:3]}
    for family_a,family_b,label in [('catboost','lgb','cat_lgb'),('tabm','lgb','tabm_lgb')]:
        left=[name for name in eligible if configs[name]['kind']==family_a]
        right=[name for name in eligible if configs[name]['kind']==family_b]
        if left and right:
            first=max(left,key=lambda name:average_precision_score(yv,scores[name]))
            second=max(right,key=lambda name:average_precision_score(yv,scores[name]))
            for weight in [.25,.5,.75]:blends[f'blend_{label}_{weight}']={first:weight,second:1-weight}
    for name,parts in blends.items():
        scores[name]=sum(scores[k]*w for k,w in parts.items())
        configs[name]={'kind':'blend','parts':parts}
        results.append(record(name,yv,scores[name]))
    winner=max([name for name in scores if name!='v1_control'],key=lambda name:average_precision_score(yv,scores[name]))
    dump(OUT/'models/configs.json',configs)
    pd.DataFrame(results).to_csv(OUT/'tuning_candidates.csv',index=False)
    predictions=mart.loc[tune_idx,['channel_id','date',TARGET]].copy()
    for name,s in scores.items():predictions[name]=s
    predictions.to_parquet(OUT/'predictions/tune.parquet',index=False)
    selection={'selected_model':winner,'criterion':'AP October-November2025','configs':configs,
               'prior_path':context['prior_path'],'intraday':context['intraday'],
               'train_rows':context['train_rows'],'features':context['features'],'tune_start':'2025-10-01','tune_end':'2025-11-29',
               'calibration_start':'2025-12-01','calibration_end':'2025-12-30','target':TARGET,
               'input_manifest_sha256':hashlib.sha256((OUT/'input_provenance.json').read_bytes()).hexdigest()}
    dump(OUT/'selection_pre_calibration.json',selection)
    smoke=tune_idx[:256]
    restored=score_config(winner,configs,mart,smoke)
    difference=float(np.max(np.abs(restored-scores[winner][:256])))
    if difference>1e-6:raise ValueError(f'Saved inference mismatch {difference}')
    dump(OUT/'inference_check.json',{'rows':256,'maximum_score_difference':difference,'model':winner})
    if choose_only:
        v1.log(f'SELECTED {winner}; calibration and test NOT evaluated')
        return
    score=score_config(winner,configs,mart,cal_idx)
    threshold,op=v1.choose_threshold(mart.loc[cal_idx,TARGET],score)
    selection.update(threshold=threshold,operating_points=op)
    dump(OUT/'selection.json',selection)
    cal=mart.loc[cal_idx,['channel_id','date',TARGET]].copy();cal['score']=score
    cal.to_parquet(OUT/'predictions/calibration.parquet',index=False)
    dump(OUT/'calibration_metrics.json',record(winner,cal[TARGET].to_numpy(),score))
    v1.log(f'SELECTED {winner}; threshold {threshold:.6f}; test NOT evaluated')


def finalize():
    from refit_components import refit_components
    selection=json.loads((OUT/'selection_pre_calibration.json').read_text())
    if selection['input_manifest_sha256']!=hashlib.sha256((OUT/'input_provenance.json').read_bytes()).hexdigest():
        raise ValueError('Input manifest changed after model selection')
    mart=load()
    if selection['prior_path']:
        mart=mart.merge(pd.read_parquet(selection['prior_path']),on='channel_id',how='left',validate='many_to_one',sort=False)
    if selection['intraday']:
        intra=pd.concat([pd.read_parquet(ROOT/f'data/interim/intraday_{y}.parquet') for y in [2025,2026]],ignore_index=True)
        mart=mart.merge(intra,on=['channel_id','date'],how='left',validate='one_to_one',sort=False)
    name=selection['selected_model']
    configs=refit_components(name,selection['configs'],mart,OUT)
    selection.update(configs=configs,refit_end='2025-11-29',refit_trees='fixed from tuning')
    # Refit graph is frozen before looking at the threshold-setting period.
    dump(OUT/'selection_refit_before_calibration.json',selection)
    idx=mart.index[(mart.date>='2025-12-01')&(mart.date<='2025-12-30')]
    score=score_config(name,configs,mart,idx)
    threshold,operating=v1.choose_threshold(mart.loc[idx,TARGET],score)
    selection.update(threshold=threshold,operating_points=operating,
        reference_targets={'precision_gt':.7,'recall_gt':.5,'official_requirement':False})
    dump(OUT/'selection.json',selection)
    cal=mart.loc[idx,['channel_id','date',TARGET]].copy();cal['score']=score
    cal.to_parquet(OUT/'predictions/calibration.parquet',index=False)
    dump(OUT/'calibration_metrics.json',record(name,cal[TARGET].to_numpy(),score))
    check=score_config(name,configs,mart,idx[:256])
    difference=float(np.max(np.abs(check-score[:256])))
    if difference>1e-6:raise ValueError('Refit inference differs between batch sizes')
    dump(OUT/'refit_inference_check.json',{'rows':256,'maximum_score_difference':difference})
    v1.log(f'Refit and threshold FROZEN: {name} {threshold:.6f}. Test not evaluated.')


def score_config(name,configs,mart,idx):
    cfg=configs[name]
    if cfg['kind']=='tabm':
        from tabm_candidate import load_predictor
        return load_predictor()(mart.loc[idx])
    if cfg['kind']=='blend':return sum(weight*score_config(k,configs,mart,idx) for k,weight in cfg['parts'].items())
    if cfg['kind']=='experts':
        score=score_config(cfg['fallback'],configs,mart,idx)
        for system,filename in cfg['specialists'].items():
            mask=mart.loc[idx,'engineering_system'].astype(str).to_numpy()==system
            if not mask.any():continue
            m=lgb.Booster(model_file=str(OUT/'models'/f'{filename}.txt'))
            score[mask]=predict(m,'lgb',mart,idx[mask],cfg['features'],cfg['cats'])
        return score
    if cfg.get('external_v1'):
        m=CatBoostClassifier();m.load_model(str(ROOT/'reports/improved_v1/models/catboost_history.cbm'))
    else:m=model_load(name,cfg)
    return predict(m,cfg['kind'],mart,idx,cfg['features'],cfg['cats'])


def evaluate():
    cfg=json.loads((OUT/'selection.json').read_text())
    if cfg['input_manifest_sha256']!=hashlib.sha256((OUT/'input_provenance.json').read_bytes()).hexdigest():
        raise ValueError('Input manifest changed after selection')
    mart=load()
    if cfg['prior_path']:
        mart=mart.merge(pd.read_parquet(cfg['prior_path']),on='channel_id',how='left',validate='many_to_one',sort=False)
    if cfg['intraday']:
        intra=pd.concat([pd.read_parquet(ROOT/f'data/interim/intraday_{y}.parquet') for y in [2025,2026]],ignore_index=True)
        mart=mart.merge(intra,on=['channel_id','date'],how='left',validate='one_to_one',sort=False)
    idx=v1.legacy.sample_eval(mart[mart.split=='test'],1200000,42).index
    pred=mart.loc[idx,['channel_id','date',TARGET,'alarm_count','events_count']].copy()
    pred['score_v2']=score_config(cfg['selected_model'],cfg['configs'],mart,idx)
    old=pd.read_parquet(ROOT/'reports/improved_v1/predictions/test.parquet')
    pred=pred.merge(old[['channel_id','date','score']],on=['channel_id','date'],validate='one_to_one').rename(columns={'score':'score_v1'})
    if len(pred)!=1200000:raise ValueError('Test identities mismatch')
    t1=json.loads((ROOT/'reports/improved_v1/selection.json').read_text())['threshold']
    rows=[{'model':'v1',**v1.metrics(pred[TARGET],pred.score_v1,t1)},
          {'model':cfg['selected_model'],**v1.metrics(pred[TARGET],pred.score_v2,cfg['threshold'])}]
    pd.DataFrame(rows).to_csv(OUT/'test_comparison.csv',index=False)
    precise=cfg['operating_points']['threshold_precision_gt_0_7']
    if precise is not None:
        dump(OUT/'test_precision_operating_point.json',v1.metrics(pred[TARGET],pred.score_v2,precise))
    pred.to_parquet(OUT/'predictions/test.parquet',index=False)
    monthly=[]
    for month,part in pred.groupby(pred.date.dt.to_period('M')):
        monthly.append({'month':str(month),'rows':len(part),'positives':int(part[TARGET].sum()),
           'v1_AP':average_precision_score(part[TARGET],part.score_v1),
           'v2_AP':average_precision_score(part[TARGET],part.score_v2)})
    pd.DataFrame(monthly).to_csv(OUT/'test_monthly.csv',index=False)
    dump(OUT/'test_done.json',{'selection_sha256':hashlib.sha256((OUT/'selection.json').read_bytes()).hexdigest(),
         'metrics':rows,'note':'Previously inspected2026 diagnostic, not a fresh blind holdout.'})
    v1.log(str(rows))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','freeze','train','select','finalize','evaluate'])
    p.add_argument('--prior',type=Path);p.add_argument('--intraday',action='store_true');p.add_argument('--choose-only',action='store_true')
    args=p.parse_args()
    for folder in [OUT,OUT/'models',OUT/'predictions']:folder.mkdir(parents=True,exist_ok=True)
    if args.phase=='prepare':prepare()
    elif args.phase=='freeze':freeze()
    elif args.phase=='train':train(args)
    elif args.phase=='select':select(args.choose_only)
    elif args.phase=='finalize':finalize()
    else:evaluate()
