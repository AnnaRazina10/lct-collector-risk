"""Score prepared features with a frozen v3 graph; never train or read labels."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier

DEFAULT_RUN=Path(__file__).resolve().parents[2]/'reports/improved_v3'


def score_features(features, run_dir):
    run_dir=Path(run_dir)
    cfg=json.loads((run_dir/'selection.json').read_text())
    def predict(name):
        node=cfg['configs'][name]
        if node['kind']=='blend':
            if not np.isclose(sum(node['parts'].values()),1):raise ValueError('Invalid mixture weights')
            return sum(w*predict(k) for k,w in node['parts'].items())
        if node['kind']=='product':return np.prod([predict(k) for k in node['parts']],axis=0)
        columns=node['features'];missing=sorted(set(columns)-set(features))
        if missing:raise ValueError(f'Missing features: {missing}')
        x=features[columns].copy();artifact=node.get('artifact_name',name)
        if node['kind']=='catboost':
            for c in node['cats']:x[c]=x[c].astype(str)
            for c in x.select_dtypes(include='number'):x[c]=x[c].replace([np.inf,-np.inf],np.nan).fillna(-999999)
            model=CatBoostClassifier();model.load_model(str(run_dir/'models'/f'{artifact}.cbm'))
            return model.predict_proba(x,thread_count=3)[:,1]
        if node['kind']!='lgb':raise ValueError('Unknown component kind')
        for c in node['cats']:x[c]=x[c].astype('category')
        model=lgb.Booster(model_file=str(run_dir/'models'/f'{artifact}.txt'))
        return model.predict(x,num_threads=3)
    score=predict(cfg['selected_model'])
    if not np.isfinite(score).all() or ((score<0)|(score>1)).any():
        raise ValueError('Invalid probability output')
    out=features[[c for c in ['channel_id','date'] if c in features]].copy()
    out['score']=score;out['warning']=score>=cfg['threshold']
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=score_features(pd.read_parquet(args.features),args.run_dir)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False)
    print(f'Scored {len(result)} rows; warnings={int(result.warning.sum())}')
