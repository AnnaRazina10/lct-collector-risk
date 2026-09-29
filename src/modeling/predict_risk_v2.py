"""Inference for the frozen v2 ensemble; accepts features, never fits or uses labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier


def score_features(features: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    run_dir=Path(run_dir)
    selection=json.loads((run_dir/'selection.json').read_text())
    configs=selection['configs']
    def predict(name):
        cfg=configs[name]
        kind=cfg['kind']
        if kind=='blend':
            if not np.isclose(sum(cfg['parts'].values()),1):raise ValueError('Invalid mixture weights')
            return sum(weight*predict(component) for component,weight in cfg['parts'].items())
        if kind=='tabm':
            from tabm_candidate import load_predictor
            return load_predictor(run_dir/'models')(features)
        if kind=='experts':
            result=predict(cfg['fallback'])
            for system,artifact in cfg['specialists'].items():
                mask=features.engineering_system.astype(str).to_numpy()==system
                if mask.any():result[mask]=component_prediction(artifact,'lgb',cfg,features.loc[mask])
            return result
        return component_prediction(cfg.get('artifact_name',name),kind,cfg,features)

    def component_prediction(artifact,kind,cfg,frame):
        columns=cfg['features'];missing=sorted(set(columns)-set(frame.columns))
        if missing:raise ValueError(f'Missing required features: {missing}')
        x=frame[columns].copy()
        if kind=='catboost':
            for col in cfg['cats']:x[col]=x[col].astype(str)
            for col in x.select_dtypes(include='number'):
                x[col]=x[col].replace([np.inf,-np.inf],np.nan).fillna(-999999)
            model=CatBoostClassifier();model.load_model(str(run_dir/'models'/f'{artifact}.cbm'))
            return model.predict_proba(x,thread_count=5)[:,1]
        for col in cfg['cats']:x[col]=x[col].astype('category')
        model=lgb.Booster(model_file=str(run_dir/'models'/f'{artifact}.txt'))
        return model.predict(x,num_threads=5)

    scores=predict(selection['selected_model'])
    if not np.isfinite(scores).all():raise ValueError('Non-finite ensemble score')
    out=features[[c for c in ['channel_id','date'] if c in features]].copy()
    out['score']=scores
    out['warning']=scores>=selection['threshold']
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,default=Path('reports/improved_v2'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    out=score_features(pd.read_parquet(args.features),args.run_dir)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(args.output,index=False)
    print(f'Scored {len(out):,} rows, warnings={int(out.warning.sum())}')
