"""Causal state dynamics and physical-context features for the same daily target."""
from __future__ import annotations
import numpy as np
import pandas as pd
from improve_baseline import ROOT, rolling_sum


def lag(a, n):
    out = np.zeros_like(a, dtype=np.float32)
    out[:, n:] = a[:, :-n]
    return out


def ewm(a, span):
    out = np.empty_like(a, dtype=np.float32)
    alpha = 2 / (span + 1)
    out[:, 0] = a[:, 0]
    for t in range(1, a.shape[1]):
        out[:, t] = alpha * a[:, t] + (1 - alpha) * out[:, t-1]
    return out


def streak(mask):
    t = np.arange(mask.shape[1])[None, :]
    last_not = np.maximum.accumulate(np.where(~mask, t, -1), axis=1)
    return np.where(mask, t - last_not, 0).astype(np.float32)


def extend(mart, prior_path=None):
    counts = mart.groupby('channel_id', observed=True, sort=False).size()
    if counts.nunique() != 1 or not mart[['channel_id','date']].equals(mart[['channel_id','date']].sort_values(['channel_id','date'])):
        raise ValueError('Expected a complete sorted daily grid')
    n, d = len(counts), int(counts.iloc[0])
    def arr(col): return mart[col].to_numpy().reshape(n,d)
    new = {}
    def add(name,value): new[name] = np.asarray(value,dtype=np.float32).ravel()
    a = arr('alarm_count') > 0
    obs = arr('events_count') > 0
    fault = arr('fault_count') > 0
    power = arr('no_power_count') > 0
    for name, mask in [('alarm',a),('observed',obs),('fault',fault),('no_power',power)]:
        add(name+'_streak',streak(mask))
        for span in [7,28]: add(name+f'_ewm_{span}',ewm(mask,span))
    add('quiet_streak',streak(~a))
    for days in [1,2,7,14,28]: add(f'alarm_lag_{days}',lag(a,days))
    for col in ['events_count','alarm_count']:
        logv = np.log1p(arr(col))
        for span in [7,28]: add(col+f'_log_ewm{span}',ewm(logv,span))
        add(col+'_change',logv-lag(logv,1))
        add(col+'_same_weekday_4w',sum(lag(logv,k) for k in [7,14,21,28])/4)
    prev = lag(a,1).astype(bool)
    known = np.ones_like(a); known[:,0] = False
    for source in [False,True]:
        exposure = rolling_sum((prev==source)&known,90)
        outcome = rolling_sum((prev==source)&a&known,90)
        add('transition_from_'+str(int(source)),(outcome+1)/(exposure+20))
    add('alarm_rate_when_observed_28d',rolling_sum(a,28)/(1+rolling_sum(obs,28)))
    add('alarm_starts_28d',rolling_sum(a & ~prev & known,28))
    add('alarm_switches_28d',rolling_sum((a!=prev)&known,28))
    add('observation_change',rolling_sum(obs,7)/7-rolling_sum(obs,28)/28)
    # Count-weighted prior numeric regime; current day excluded from baseline.
    num=arr('numeric_count').astype(float)
    total=arr('value_sum').astype(float)
    squares=arr('value_sumsq').astype(float)
    old_n=lag(rolling_sum(num,28),1)
    old_sum=lag(rolling_sum(total,28),1)
    old_sq=lag(rolling_sum(squares,28),1)
    old_mean=old_sum/np.maximum(old_n,1)
    old_sd=np.sqrt(np.maximum(old_sq/np.maximum(old_n,1)-old_mean**2,0))
    current=total/np.maximum(num,1)
    add('numeric_past28_mean',np.where(old_n>0,old_mean,np.nan))
    add('numeric_past28_std',np.where(old_n>0,old_sd,np.nan))
    add('numeric_deviation',np.where((num>0)&(old_n>0),np.clip((current-old_mean)/(1+old_sd),-100,100),np.nan))
    objects=pd.read_csv(ROOT/'data/raw/справочник_объектов_диспетчер.csv',dtype=str)
    parent_map=objects.set_index('ид_объект')['родитель'].fillna('root').to_dict()
    parents=mart.object_id.astype(str).map(parent_map).fillna('unknown')
    temp=pd.DataFrame({'date':mart.date,'object':mart.object_id,'sensor':mart.sensor_type,
                       'parent':parents,'alarm':a.ravel(),'observed':obs.ravel(),
                       'fault':fault.ravel(),'power':power.ravel()})
    for group, keys in [('type_object',['object','sensor','date']),('parent',['parent','date'])]:
        g=temp.groupby(keys,observed=True)
        for col in ['alarm','fault','power']:
            other=g[col].transform('sum').to_numpy()-temp[col].to_numpy(dtype=int)
            total_obs=g['observed'].transform('sum').to_numpy()-obs.ravel()
            add(f'{group}_{col}_others',other)
            add(f'{group}_{col}_rate',other/(1+total_obs))
    result=pd.concat([mart,pd.DataFrame(new,index=mart.index)],axis=1)
    if prior_path is not None:
        prior=pd.read_parquet(prior_path)
        if prior.channel_id.duplicated().any(): raise ValueError('Duplicate historical prior')
        result=result.merge(prior,on='channel_id',how='left',validate='many_to_one',sort=False)
    return result
