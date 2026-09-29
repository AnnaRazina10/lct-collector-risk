"""Bounded, temporal comparison of registered alarm onset forecasts at D+2."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

import object_risk_probe as base
from object_onset_statistics import OnsetLogistic, SmoothedOnsetFrequency
from onset_evaluation import summarize, daily_topk, paired_week_bootstrap

ROOT = base.ROOT
OUT = ROOT/'reports/object_onset_experiment'
MODELS = ROOT/'models/object_onset_experiment'
OLD_CFG = ROOT/'reports/object_risk_probe/configuration.json'
COMBINED_CFG = ROOT/'reports/object_combined/selection.json'
CATS = ['object_id','object_kind','parent_id']
EXTRA = ['onset_today','clear_today','onset_mean7','onset_mean28','clear_mean7','clear_mean28',
         'past_onset_frequency','days_since_onset','alarm_pattern3']


def onset_frame(mart):
    """Exact calendar labels plus causal past-only transition features."""
    frame = mart.sort_values(['object_id','date']).reset_index(drop=True).copy()
    if frame.duplicated(['object_id','date']).any():
        raise ValueError('Duplicate object-day')
    group = frame.groupby('object_id',observed=True,sort=False)
    previous_date = group.date.shift(1)
    if not (frame.date-previous_date).dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError('Incomplete object calendar')
    before = group.alarm_today.shift(1)
    frame['onset_today'] = (frame.alarm_today.eq(1)&before.eq(0)).astype(float).where(before.notna())
    frame['clear_today'] = (frame.alarm_today.eq(0)&before.eq(1)).astype(float).where(before.notna())
    group = frame.groupby('object_id',observed=True,sort=False)
    for prefix,column in [('onset','onset_today'),('clear','clear_today')]:
        for window in [7,28]:
            frame[f'{prefix}_mean{window}'] = group[column].transform(lambda s:s.rolling(window,min_periods=1).mean())
    frame['past_onset_frequency'] = group.onset_today.transform(lambda s:s.expanding(min_periods=1).mean())
    def age(series):
        pos=np.arange(len(series)); last=np.maximum.accumulate(np.where(series.eq(1),pos,-1))
        return np.where(last<0,-1,pos-last)
    frame['days_since_onset'] = group.onset_today.transform(age)
    frame['alarm_pattern3'] = frame.alarm_today+2*before+4*group.alarm_today.shift(2)
    future1 = group.alarm_today.shift(-1)
    future2 = group.alarm_today.shift(-2)
    known = future1.notna()&future2.notna()
    frame['quiet_intermediate'] = future1.eq(0).astype(float).where(known)
    frame['alarm_target'] = future2.where(known)
    frame['onset_target'] = (future1.eq(0)&future2.eq(1)).astype(float).where(known)
    frame['joint_target'] = (2*future1+future2).where(known)
    if 'observed_today' in frame:
        frame['records_intermediate'] = group.observed_today.shift(-1).where(known)
        frame['records_target'] = group.observed_today.shift(-2).where(known)
    return frame


def prepare(end):
    mart,_,coverage = base.prepare(end)
    return onset_frame(mart),coverage


def features():
    return json.loads(OLD_CFG.read_text())['features']+EXTRA


def metric_row(name, frame, score, k=10):
    return {'model':name,'k':k,**summarize(frame,score,k=k)}


def fit_candidate(name,train,tune=None,iterations=None):
    artifact={'name':name,'features':features()}
    if name in ['logistic','frequency']:
        artifact['kind']='statistical'
        artifact['estimator']=(OnsetLogistic() if name=='logistic' else SmoothedOnsetFrequency()).fit(train)
        return artifact
    kind,leaves=name.split('_'); leaves=int(leaves)
    artifact.update(kind=kind,models=[],iterations=[])
    specifications = {'direct':[('onset_target',None)],
                      'hurdle':[('quiet_intermediate',None),('alarm_target','quiet_intermediate')],
                      'joint':[('joint_target',None)]}[kind]
    for i,(target,condition) in enumerate(specifications):
        selected=train if condition is None else train.loc[train[condition].eq(1)]
        validation=tune if tune is None or condition is None else tune.loc[tune[condition].eq(1)]
        params=dict(objective='multiclass' if kind=='joint' else 'binary',
                    metric='multi_logloss' if kind=='joint' else 'average_precision',
                    n_estimators=iterations[i] if iterations is not None else 500,
                    num_leaves=leaves,learning_rate=.04,min_child_samples=40,reg_lambda=10,
                    colsample_bytree=.85,random_state=84,n_jobs=2,verbosity=-1)
        if kind=='joint':params['num_class']=4
        model=lgb.LGBMClassifier(**params)
        options={'categorical_feature':CATS}
        if validation is not None:
            options.update(eval_set=[(validation[artifact['features']],validation[target].astype(int))],
                           callbacks=[lgb.early_stopping(40,verbose=False),lgb.log_evaluation(0)])
        model.fit(selected[artifact['features']],selected[target].astype(int),**options)
        artifact['models'].append(model)
        artifact['iterations'].append(int(model.best_iteration_ or model.n_estimators))
    return artifact


def predict(artifact,frame):
    # No outcome is selected; statistical estimators also own an explicit feature schema.
    if artifact['kind']=='statistical':return artifact['estimator'].predict(frame)
    x=frame[artifact['features']]
    values=[m.predict_proba(x,num_threads=2) for m in artifact['models']]
    if artifact['kind']=='joint':
        column=list(artifact['models'][0].classes_).index(1)
        return values[0][:,column]
    if artifact['kind']=='hurdle':return values[0][:,1]*values[1][:,1]
    return values[0][:,1]


def save_artifact(artifact,stage):
    path=MODELS/stage/f'{artifact["name"]}.joblib';path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(artifact,path)
    return {'path':str(path.relative_to(ROOT)),'sha256':base.checksum(path),
            'iterations':artifact.get('iterations'),'kind':artifact['kind']}


def load_artifact(node):
    path=ROOT/node['path']
    if base.checksum(path)!=node['sha256']:raise ValueError('Saved candidate checksum changed')
    return joblib.load(path)


def source_hashes():
    paths=[OUT/'protocol.md',Path(__file__),ROOT/'src/modeling/object_risk_probe.py',
           ROOT/'src/modeling/object_onset_statistics.py',ROOT/'src/modeling/onset_evaluation.py',OLD_CFG,
           base.CACHE/'daily_2025.parquet',ROOT/'data/raw/справочник_каналов_датчиков.csv',
           ROOT/'data/raw/справочник_объектов_диспетчер.csv']
    return {str(p.relative_to(ROOT)):base.checksum(p) for p in paths}


def verify_sources(cfg):
    for path,digest in cfg['sources'].items():
        if base.checksum(ROOT/path)!=digest:raise ValueError(f'Frozen source changed: {path}')


def baseline_score(frame,refit=False):
    if refit:
        node=json.loads(COMBINED_CFG.read_text())['components']['lightgbm_control']
        path=ROOT/node['path']; expected=node['sha256']; columns=node['features']
    else:
        cfg=json.loads(OLD_CFG.read_text());path=ROOT/cfg['model_path'];expected=cfg['model_sha256'];columns=cfg['features']
    if base.checksum(path)!=expected:raise ValueError('Any-alarm control changed')
    return lgb.Booster(model_file=str(path)).predict(frame[columns],num_threads=2)


def select():
    if (OUT/'selection.json').exists():raise FileExistsError('Selection already frozen; do not silently retune')
    start=time.monotonic();mart,coverage=prepare('2025-11-30')
    train=mart.loc[mart.date.le('2025-09-28')].copy()
    tune=mart.loc[mart.date.between('2025-10-01','2025-11-28')].copy()
    assert train.onset_target.notna().all() and tune.onset_target.notna().all()
    names=[f'{kind}_{leaves}' for kind in ['direct','hurdle','joint'] for leaves in [7,15]]+['logistic','frequency']
    rows=[];scores={};nodes={}
    for name in names:
        artifact=fit_candidate(name,train,tune)
        scores[name]=predict(artifact,tune)
        nodes[name]=save_artifact(artifact,'tune')
        rows.append(metric_row(name,tune,scores[name]));print(json.dumps(rows[-1]),flush=True)
    def order(name):
        row=next(r for r in rows if r['model']==name)
        return (-row['precision_at_k'],-row['average_precision'],name)
    tree=min(names[:6],key=order);stat=min(names[6:],key=order)
    parts={name:{name:1.} for name in names}
    for weight in [.25,.5,.75]:
        name=f'blend_tree{weight:.2f}'
        parts[name]={tree:weight,stat:1-weight}
        scores[name]=weight*scores[tree]+(1-weight)*scores[stat]
        rows.append(metric_row(name,tune,scores[name]))
    winner=min(scores,key=order)
    rows.append(metric_row('previous_any_alarm',tune,baseline_score(tune)))
    pd.DataFrame(rows).to_csv(OUT/'tuning_metrics.csv',index=False)
    cfg={'selected':winner,'parts':parts[winner],'best_tree':tree,'best_stat':stat,'candidates':nodes,
         'features':features(),'primary_k':10,'criterion':'Oct-Nov Precision@10, then AP, then name',
         'train_feature_end':'2025-09-28','train_label_end':'2025-09-30',
         'tune_feature_start':'2025-10-01','tune_feature_end':'2025-11-28','tune_label_end':'2025-11-30',
         'december_read':False,'2026_read':False,'sources':source_hashes(),'coverage':coverage,
         'elapsed_seconds':time.monotonic()-start}
    base.dump(OUT/'selection.json',cfg)
    prediction=tune[['object_id','date','onset_target','alarm_target']].copy()
    for name,value in scores.items():prediction[name]=value
    prediction.to_parquet(base.CACHE/'object_onset_tune.parquet',index=False)
    print(json.dumps({'selected':winner,'parts':parts[winner]},ensure_ascii=False),flush=True)


def component_scores(frame,cfg):
    scores={name:predict(load_artifact(node),frame) for name,node in cfg['models'].items()}
    scores['selected']=sum(weight*scores[name] for name,weight in cfg['parts'].items())
    scores['any_alarm_refit']=baseline_score(frame,refit=True)
    scores['any_alarm_previous']=baseline_score(frame)
    return scores


def refit():
    if (OUT/'refit.json').exists():raise FileExistsError('Refit already frozen')
    selection=json.loads((OUT/'selection.json').read_text());verify_sources(selection)
    mart,_=prepare('2025-12-31');train=mart.loc[mart.date.le('2025-11-28')]
    december=mart.loc[mart.date.between('2025-12-01','2025-12-29')]
    names=sorted(set([selection['best_tree'],selection['best_stat'],*selection['parts']]))
    nodes={}
    for name in names:
        model=fit_candidate(name,train,iterations=selection['candidates'][name]['iterations'])
        nodes[name]=save_artifact(model,'refit')
    cfg={'models':nodes,'parts':selection['parts'],'selection_sha256':base.checksum(OUT/'selection.json'),
         'sources':selection['sources'],'feature_end':'2025-11-28','label_end':'2025-11-30',
         'any_alarm_refit_configuration_sha256':base.checksum(COMBINED_CFG),'2026_read':False}
    values=component_scores(december,cfg)
    rows=[metric_row(name,december,score) for name,score in values.items()]
    pd.DataFrame(rows).to_csv(OUT/'december_metrics.csv',index=False)
    byname={r['model']:r for r in rows}
    cfg['december_gate_passed']=byname['selected']['precision_at_k']>=byname['any_alarm_refit']['precision_at_k']
    cfg['gate']='December Precision@10 >= comparable any-alarm control; no tuning on December'
    base.dump(OUT/'refit.json',cfg)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


def evaluate():
    cfg=json.loads((OUT/'refit.json').read_text());verify_sources(cfg)
    if base.checksum(OUT/'selection.json')!=cfg['selection_sha256'] or base.checksum(COMBINED_CFG)!=cfg['any_alarm_refit_configuration_sha256']:
        raise ValueError('Frozen configuration changed')
    mart,coverage=prepare('2026-06-30')
    test=mart.loc[mart.date.between('2025-12-31','2026-06-28')].copy()
    assert len(test)==14040 and test.onset_target.sum()==2000 and test.alarm_target.sum()==4203
    values=component_scores(test,cfg)
    rows=[metric_row(name,test,score,k) for k in [5,10,20] for name,score in values.items()]
    pd.DataFrame(rows).to_csv(OUT/'evaluation_metrics.csv',index=False)
    monthly=[]
    for month,part in test.groupby((test.date+pd.Timedelta(days=2)).dt.strftime('%Y-%m')):
        positions=test.index.get_indexer(part.index)
        for name,score in values.items():monthly.append({'target_month':month,**metric_row(name,part,score[positions])})
    pd.DataFrame(monthly).to_csv(OUT/'monthly_metrics.csv',index=False)
    strata=[]
    for name,score in values.items():
        warnings=daily_topk(test,score,k=10)
        for observed in [0,1]:
            mask=test.records_intermediate.eq(observed).to_numpy()
            positive=test.onset_target.eq(1).to_numpy()
            tp=int((warnings&positive&mask).sum());alerts=int((warnings&mask).sum());onsets=int((positive&mask).sum())
            strata.append({'model':name,'records_on_intermediate_day':observed,'rows':int(mask.sum()),
                           'onsets':onsets,'warnings':alerts,'true_onset_warnings':tp,
                           'precision_within_original_warnings':tp/max(alerts,1),'recall_within_stratum':tp/max(onsets,1),
                           'retrospective_stratum_not_selection_filter':True})
    pd.DataFrame(strata).to_csv(OUT/'observability.csv',index=False)
    uncertainty=paired_week_bootstrap(test,values['selected'],values['any_alarm_refit'],k=10,n=1000,seed=84)
    base.dump(OUT/'uncertainty.json',uncertainty)
    # Inference from a strictly shorter history, never from future label columns.
    prefix,_=prepare('2026-03-15')
    earlier=test.loc[test.date.le('2026-03-15')]
    truncated=prefix.loc[prefix.date.between('2025-12-31','2026-03-15')].copy()
    assert earlier[['object_id','date']].reset_index(drop=True).equals(truncated[['object_id','date']].reset_index(drop=True))
    drop=[base.TARGET,'onset_target','quiet_intermediate','alarm_target','joint_target','records_intermediate','records_target']
    prefix_values=component_scores(truncated.drop(columns=drop),cfg)
    positions=test.index.get_indexer(earlier.index)
    deltas={name:float(np.max(np.abs(prefix_values[name]-score[positions]))) for name,score in values.items()}
    assert max(deltas.values())<1e-12
    final=mart.loc[mart.date.eq('2026-06-28')]
    latest,_=prepare('2026-06-28');latest=latest.loc[latest.date.eq('2026-06-28')]
    latest_values=component_scores(latest.drop(columns=drop),cfg)
    full_values=component_scores(final.drop(columns=drop),cfg)
    last_deltas={name:float(np.max(np.abs(latest_values[name]-full_values[name]))) for name in latest_values}
    assert max(last_deltas.values())<1e-12
    prediction=test[['object_id','date','onset_target','alarm_target','observed_today']].copy()
    for name,score in values.items():prediction[name]=score
    path=base.CACHE/'object_onset_test.parquet';prediction.to_parquet(path,index=False)
    base.dump(OUT/'checks.json',{'rows':len(test),'objects':test.object_id.nunique(),'onsets':int(test.onset_target.sum()),
        'any_alarm_days':int(test.alarm_target.sum()),'primary_k':10,'test_is_blind':False,
        'prefix_max_difference':deltas,'latest_78_without_future_or_targets_difference':last_deltas,
        'refit_configuration_sha256':base.checksum(OUT/'refit.json'),'predictions_sha256':base.checksum(path),
        'daily_2026_sha256':base.checksum(base.CACHE/'daily_2026.parquet'),'coverage':coverage,
        'december_gate_passed':cfg['december_gate_passed'],'production_model_replaced':False})
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print(json.dumps(uncertainty),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['select','refit','evaluate'])
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'protocol.md').exists():raise FileNotFoundError('Write protocol before experiment')
    globals()[args.phase]()
