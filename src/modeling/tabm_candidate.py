"""Bounded official TabM candidate; train/tuning periods end before December 2025.

Uses https://github.com/yandex-research/tabm (Apache-2.0), not a reimplementation.
Train-time preprocessing and category vocabularies are saved for later inference.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/improved_v2'
TARGET = 'target_alarm_next_24h'
EXCLUDED = {'date', 'channel_id', TARGET, 'split', 'value_sum', 'value_sumsq', 'year', 'month'}
CATEGORICAL = ['engineering_system', 'sensor_type', 'object_id', 'object_kind', 'channel_category']
TRAIN_END = pd.Timestamp('2025-09-29')
TUNE_START = pd.Timestamp('2025-10-01')
TUNE_END = pd.Timestamp('2025-11-29')


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def numeric_values(frame, columns):
    values = frame[columns].to_numpy(dtype=np.float32, copy=True)
    values[~np.isfinite(values)] = np.nan
    # Robust monotone compression before standardisation; NaNs stay missing.
    return np.sign(values) * np.log1p(np.abs(values))


def fit_preprocessing(train, features, max_categories=512):
    cats = [c for c in CATEGORICAL if c in features]
    nums = [c for c in features if c not in cats]
    for c in nums:
        if not pd.api.types.is_numeric_dtype(train[c]):
            raise ValueError(f'Non-numeric feature outside categorical allowlist: {c}')
    imputer = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
    numeric = imputer.fit_transform(numeric_values(train, nums)).astype(np.float32)
    scaler = StandardScaler().fit(numeric)
    maps = {}
    for col in cats:
        values = train[col].astype('string').fillna('<MISSING>')
        # Native TabM uses one-hot categories. Cap high-cardinality identifiers
        # using train frequencies only; unseen/rare categories map to zero.
        counts = values.value_counts()
        ordered = sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))
        maps[col] = {str(value): i + 1 for i, (value, _) in enumerate(ordered[:max_categories])}
    return {'features': features, 'numeric': nums, 'categorical': cats,
            'imputer': imputer, 'scaler': scaler, 'maps': maps,
            'max_categories': max_categories, 'n_numeric_transformed': numeric.shape[1]}


def transform(frame, prep):
    missing = sorted(set(prep['features']) - set(frame.columns))
    if missing:
        raise ValueError(f'Missing inference features: {missing}')
    num = prep['imputer'].transform(numeric_values(frame, prep['numeric']))
    num = prep['scaler'].transform(num).astype(np.float32)
    num = np.clip(num, -12, 12)
    cat = np.column_stack([
        frame[c].astype('string').fillna('<MISSING>').map(prep['maps'][c]).fillna(0).to_numpy(np.int64)
        for c in prep['categorical']
    ])
    if not np.isfinite(num).all():
        raise ValueError('Non-finite transformed inputs')
    return num, cat


def create_model(config, device):
    from tabm import TabM
    return TabM.make(**config).to(device)


def predict_arrays(model, num, cat, device, batch_size=2048):
    import torch
    model.eval()
    scores = np.empty(len(num), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(num), batch_size):
            stop = min(start + batch_size, len(num))
            x = torch.as_tensor(num[start:stop], device=device)
            c = torch.as_tensor(cat[start:stop], device=device)
            # Each ensemble member emits its own binary logit; average probabilities.
            scores[start:stop] = model(x, c).squeeze(-1).sigmoid().mean(1).cpu().numpy()
    return scores


def load_predictor(model_dir=None, device='cpu', threads=3):
    """Return a frame->probability callable without refitting any preprocessing."""
    import torch
    torch.set_num_threads(threads)
    directory = Path(model_dir or OUT / 'models')
    meta = json.loads((directory / 'tabm_config.json').read_text())
    with (directory / 'tabm_preprocessing.pkl').open('rb') as handle:
        prep = pickle.load(handle)  # Local artifact produced by this script only.
    model = create_model(meta['model_config'], device)
    model.load_state_dict(torch.load(directory / 'tabm.pt', map_location=device, weights_only=True))

    def predict(frame):
        num, cat = transform(frame, prep)
        return predict_arrays(model, num, cat, device)

    return predict


def summarize(y, scores):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    idx = int(np.argmax(f1))
    return {'ap': float(average_precision_score(y, scores)),
            'max_f1': float(f1[idx]), 'precision_at_max_f1': float(precision[idx]),
            'recall_at_max_f1': float(recall[idx]), 'tune_f1_threshold': float(thresholds[idx]),
            'recall_at_precision_0_7': float(np.max(recall[precision >= .7], initial=0)),
            'precision_at_recall_0_5': float(np.max(precision[recall >= .5], initial=0)),
            'rows': len(y), 'positives': int(np.sum(y))}


def run(args):
    import torch
    from torch.nn import functional as F
    start = time.monotonic()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(4252)
    rng = np.random.default_rng(4252)
    device = args.device
    if device == 'auto':
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    log(f'TabM device={device}; torch={torch.__version__}; threads={args.threads}')
    model_dir = OUT / 'models'
    pred_dir = OUT / 'predictions'
    model_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    path = ROOT / 'data/interim/advanced_mart.parquet'
    # Filters deliberately prevent reading December targets or any 2026 labels.
    train_all = pd.read_parquet(path, filters=[('date', '<=', TRAIN_END)])
    assert train_all.date.max() <= TRAIN_END
    labels_all = train_all[TARGET].to_numpy()
    pos = np.flatnonzero(labels_all == 1)
    neg = np.flatnonzero(labels_all == 0)
    if len(pos) >= args.train_rows:
        pos_keep = rng.choice(pos, args.train_rows // 4, replace=False)
    else:
        pos_keep = pos
    neg_keep = rng.choice(neg, min(len(neg), args.train_rows-len(pos_keep)), replace=False)
    idx = np.r_[pos_keep, neg_keep]
    weight = np.r_[np.full(len(pos_keep), len(pos)/len(pos_keep)),
                   np.full(len(neg_keep), len(neg)/len(neg_keep))].astype(np.float32)
    order = rng.permutation(len(idx))
    idx, weight = idx[order], weight[order]
    weight /= weight.mean()
    original_prior = float(labels_all.mean())
    train = train_all.iloc[idx].copy()
    original_rows = len(train_all)
    del train_all, labels_all
    gc.collect()
    features = [c for c in train.columns if c not in EXCLUDED]
    prep = fit_preprocessing(train, features, args.max_categories)
    num, cat = transform(train, prep)
    y = train[TARGET].to_numpy(np.float32)
    del train
    gc.collect()
    tune = pd.read_parquet(path, filters=[('date', '>=', TUNE_START), ('date', '<=', TUNE_END)])
    assert tune.date.min() >= TUNE_START and tune.date.max() <= TUNE_END
    tune_keys = tune[['channel_id', 'date', TARGET]].copy()
    tune_num, tune_cat = transform(tune, prep)
    tune_y = tune[TARGET].to_numpy()
    del tune
    gc.collect()
    early_idx = np.sort(rng.choice(len(tune_y), min(args.early_rows, len(tune_y)), replace=False))
    early_num, early_cat, early_y = tune_num[early_idx], tune_cat[early_idx], tune_y[early_idx]
    # Small native TabM, all parameters explicit for reproducible inference.
    config = {'n_num_features': num.shape[1],
              'cat_cardinalities': [len(prep['maps'][c]) + 1 for c in prep['categorical']],
              'd_out': 1, 'k': 8, 'n_blocks': 2, 'd_block': 128, 'dropout': 0.1,
              'arch_type': 'tabm-mini'}
    model = create_model(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-4)
    x_tensor = torch.as_tensor(num, device=device)
    c_tensor = torch.as_tensor(cat, device=device)
    y_tensor = torch.as_tensor(y, device=device)
    w_tensor = torch.as_tensor(weight, device=device)
    metadata = {'model_config': config, 'features': features,
                'train_end': str(TRAIN_END.date()), 'tune_start': str(TUNE_START.date()),
                'tune_end': str(TUNE_END.date()), 'train_rows_full': original_rows,
                'train_rows_sample': len(y), 'train_positives_sample': int(y.sum()),
                'train_prior_full': original_prior, 'weighting': 'inverse inclusion probability, mean-normalized',
                'early_stopping_rows': len(early_idx), 'tune_rows': len(tune_y),
                'category_cap': args.max_categories, 'category_unknown_code': 0,
                'numeric_transform': 'signed_log1p; train median impute+missing indicators; train StandardScaler; clip[-12,12]',
                'device': device, 'versions': {p: importlib.metadata.version(p) for p in ['torch','tabm','numpy','pandas','scikit-learn']},
                'source': 'https://github.com/yandex-research/tabm', 'seed': 4252,
                'test_accessed': False, 'epochs': []}
    log(f'Prepared train={len(y):,}, tune={len(tune_y):,}, dimensions={num.shape[1]}, categories={config["cat_cardinalities"]}')
    best_ap, best_state, best_epoch, stale = -math.inf, None, 0, 0
    training_start = time.monotonic()
    for epoch in range(1, args.epochs+1):
        if best_state is not None and time.monotonic() - training_start > args.seconds:
            log('Training budget reached')
            break
        epoch_start = time.monotonic()
        model.train()
        permutation = torch.randperm(len(y), device=device)
        loss_sum = 0.0
        completed_rows = 0
        for batch in permutation.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_tensor[batch], c_tensor[batch]).squeeze(-1)
            losses = F.binary_cross_entropy_with_logits(logits, y_tensor[batch, None].expand_as(logits), reduction='none')
            loss = (losses.mean(1) * w_tensor[batch]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(batch)
            completed_rows += len(batch)
            if time.monotonic() - training_start >= args.seconds:
                log('Budget reached within epoch; evaluating the current checkpoint')
                break
        scores = predict_arrays(model, early_num, early_cat, device)
        ap = float(average_precision_score(early_y, scores))
        row = {'epoch': epoch, 'early_ap': ap, 'weighted_logloss': loss_sum/completed_rows,
               'completed_rows': completed_rows, 'full_epoch': completed_rows == len(y),
               'seconds': time.monotonic()-epoch_start}
        metadata['epochs'].append(row)
        log(json.dumps(row))
        if ap > best_ap:
            best_ap, best_epoch, stale = ap, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            log('Early stopping')
            break
    if best_state is None:
        raise RuntimeError('No completed epoch')
    model.load_state_dict(best_state)
    prediction_start = time.monotonic()
    tune_scores = predict_arrays(model, tune_num, tune_cat, device)
    metrics = summarize(tune_y, tune_scores)
    metrics.update({'best_epoch': best_epoch, 'best_early_ap': best_ap,
                    'training_seconds': prediction_start-training_start,
                    'full_tune_inference_seconds': time.monotonic()-prediction_start,
                    'total_seconds': time.monotonic()-start, 'device': device,
                    'note': 'Tuning-set metrics only; no December calibration or 2026 test was accessed.'})
    torch.save(best_state, model_dir / 'tabm.pt')
    with (model_dir / 'tabm_preprocessing.pkl').open('wb') as handle:
        pickle.dump(prep, handle)
    metadata['best_epoch'] = best_epoch
    (model_dir / 'tabm_config.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    tune_keys['score'] = tune_scores
    tune_keys.to_parquet(pred_dir / 'tabm_tune.parquet', index=False)
    (OUT / 'tabm_metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    # Verify saved weights reproduce the in-memory native model on held-out tune rows.
    restored = create_model(config, device)
    restored.load_state_dict(torch.load(model_dir / 'tabm.pt', map_location=device, weights_only=True))
    check = predict_arrays(restored, tune_num[:256], tune_cat[:256], device)
    if not np.allclose(check, tune_scores[:256], atol=1e-6):
        raise AssertionError('Saved TabM predictions differ')
    log('COMPLETE ' + json.dumps(metrics))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-rows', type=int, default=300000)
    parser.add_argument('--early-rows', type=int, default=120000)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--seconds', type=float, default=540)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--max-categories', type=int, default=512)
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--device', default='auto', choices=['auto','cpu','mps'])
    run(parser.parse_args())
