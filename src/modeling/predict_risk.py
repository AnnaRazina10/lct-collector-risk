#!/usr/bin/env python3
"""Score a feature table with the frozen selected model; no retraining or labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier


def score_features(features: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    config = json.loads((run_dir / "selection.json").read_text())
    columns = config["features"]
    missing = sorted(set(columns) - set(features))
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    x = features[columns].copy()
    name = config["selected_model"]
    if name.startswith("catboost"):
        for col in config["categorical_features"]:
            x[col] = x[col].astype(str)
        for col in x.select_dtypes(include="number"):
            x[col] = x[col].replace([np.inf, -np.inf], np.nan).fillna(-999999)
        model = CatBoostClassifier()
        model.load_model(str(run_dir / "models" / f"{name}.cbm"))
        score = model.predict_proba(x, thread_count=6)[:, 1]
    else:
        for col in config["categorical_features"]:
            x[col] = x[col].astype("category")
        model = lgb.Booster(model_file=str(run_dir / "models" / f"{name}.txt"))
        score = model.predict(x, num_threads=6)
    if not np.isfinite(score).all():
        raise ValueError("Model returned non-finite scores")
    output = features[[c for c in ["channel_id", "date"] if c in features]].copy()
    output["score"] = score
    output["warning"] = score >= config["threshold"]
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=Path("reports/improved_v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score_features(pd.read_parquet(args.features), args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Scored {len(result):,} rows; warnings={int(result.warning.sum())}")
