"""Bounded, causal object-day GRU experiment; no December/2026 model selection."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn

import object_risk_probe as probe

ROOT = probe.ROOT
OUT = ROOT / 'reports/object_sequence_candidate'
TARGET = probe.TARGET
BINARY_COLUMNS = ['alarm_today', 'observed_today']
COUNT_COLUMNS = ['alarm_count', 'observed_channels', 'fault_count', 'no_power_count', 'events_count']
TRAIN_END = pd.Timestamp('2025-09-28')
TUNE_START = pd.Timestamp('2025-10-01')
TUNE_END = pd.Timestamp('2025-11-28')
LAST_NEEDED_DAY = pd.Timestamp('2025-11-30')
SEED = 84
CONFIG = {'hidden_size': 32, 'head_size': 16, 'layers': 1, 'learning_rate': .001,
          'weight_decay': .0001, 'batch_size': 256, 'max_epochs': 40,
          'patience': 6, 'minimum_ap_improvement': .0001, 'seconds_per_candidate': 240}


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1 << 20), b''):
            digest.update(part)
    return digest.hexdigest()


def log_counts(frame):
    x = frame[COUNT_COLUMNS].to_numpy(dtype=np.float32)
    if not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError('Sequence counts must be finite and non-negative')
    return np.log1p(x)


def fit_scaler(frame, cutoff=TRAIN_END):
    train = frame.loc[frame.date.le(cutoff)]
    if train.empty:
        raise ValueError('No training rows available for scaler')
    x = log_counts(train)
    mean = x.mean(axis=0, dtype=np.float64)
    std = x.std(axis=0, dtype=np.float64)
    std[std < 1e-8] = 1
    return {'columns': COUNT_COLUMNS, 'mean': mean.tolist(), 'std': std.tolist(),
            'fit_rows': len(train), 'fit_first_day': str(train.date.min().date()),
            'fit_last_day': str(train.date.max().date())}


def daily_features(frame, scaler):
    binary = frame[BINARY_COLUMNS].to_numpy(dtype=np.float32)
    if not np.isin(binary, [0, 1]).all():
        raise ValueError('Alarm and observation inputs must be binary')
    counts = (log_counts(frame)-np.asarray(scaler['mean']))/np.asarray(scaler['std'])
    # Last channel marks actual history: zero padding differs from a silent day.
    return np.column_stack([binary, counts, np.ones(len(frame))]).astype(np.float32)


def build_windows(frame, scaler, length):
    """Position i contains only its own object's days [D-length+1, D]."""
    if length < 1 or not frame.index.equals(pd.RangeIndex(len(frame))):
        raise ValueError('Positive window length and contiguous positional index required')
    if frame[['object_id', 'date']].duplicated().any():
        raise ValueError('Duplicate object/day keys')
    values = daily_features(frame, scaler)
    windows = np.zeros((len(frame), length, values.shape[1]), dtype=np.float32)
    for _, part in frame.groupby('object_id', observed=True, sort=False):
        dates = part.date.to_numpy(dtype='datetime64[D]')
        if len(dates) > 1 and not np.all(np.diff(dates) == np.timedelta64(1, 'D')):
            raise ValueError('Each object history must be sorted and daily-contiguous')
        positions = part.index.to_numpy()
        padded = np.pad(values[positions], ((length-1, 0), (0, 0)))
        windows[positions] = np.lib.stride_tricks.sliding_window_view(padded, length, axis=0).transpose(0, 2, 1)
    if not np.array_equal(windows[:, -1, :], values):
        raise AssertionError('Sequence must end on its feature day')
    return windows


class ObjectGRU(nn.Module):
    def __init__(self, input_size=8, hidden_size=32, head_size=16, layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, head_size), nn.ReLU(), nn.Linear(head_size, 1))

    def forward(self, sequence):
        _, state = self.gru(sequence)
        return self.head(state[-1]).squeeze(-1)


def predict_tensor(model, sequences, batch_size=512):
    model.eval()
    scores = []
    with torch.inference_mode():
        for batch in sequences.split(batch_size):
            scores.append(model(batch).sigmoid().numpy())
    return np.concatenate(scores)


def load_predictor(checkpoint_path, threads=3):
    """Return history-frame predictor; supplied row indexes are positional."""
    if not 1 <= threads <= 3:
        raise ValueError('Use between one and three CPU threads')
    torch.set_num_threads(threads)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = ObjectGRU(**checkpoint['architecture'])
    model.load_state_dict(checkpoint['state_dict'])

    def predict(frame, indices=None):
        windows = build_windows(frame, checkpoint['scaler'], checkpoint['window_days'])
        if indices is not None:
            windows = windows[indices]
        return predict_tensor(model, torch.from_numpy(windows))

    return predict


def complete_epoch_budget(checkpoint, checkpoint_path):
    """Reject a partial winning epoch, including legacy checkpoints without a flag."""
    complete = checkpoint.get('best_epoch_full_epoch')
    if complete is None:
        manifest_path = Path(checkpoint_path).parent.parent / 'configuration.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            for candidate in manifest.get('candidates', []):
                if candidate.get('model_sha256') == checksum(checkpoint_path):
                    complete = next((epoch['full_epoch'] for epoch in candidate['history']
                                     if epoch['epoch'] == checkpoint['best_epoch']), False)
                    break
    if complete is not True or int(checkpoint['best_epoch']) < 1:
        raise ValueError('Fixed-epoch refit requires a verified complete winning epoch')
    return int(checkpoint['best_epoch'])


def refit_fixed_epochs(history_frame, checkpoint_path, feature_end, label_end, output_path, threads=3):
    """Optional caller-invoked refit, with no validation or date selection.

    Supply daily history through label_end (at least feature_end + 2 days).
    Only alarm_today on D+2 is used for labels; features/scaler stop at feature_end.
    No data are read here, and the tuned checkpoint is never overwritten.
    """
    source_path, destination = Path(checkpoint_path), Path(output_path)
    if destination.resolve() == source_path.resolve() or destination.exists():
        raise ValueError('Refit requires a new artifact path')
    if not 1 <= threads <= 3:
        raise ValueError('Use between one and three CPU threads')
    torch.set_num_threads(threads)
    checkpoint = torch.load(source_path, map_location='cpu', weights_only=True)
    epochs = complete_epoch_budget(checkpoint, source_path)
    cutoff, label_cutoff = pd.Timestamp(feature_end), pd.Timestamp(label_end)
    if cutoff + pd.Timedelta(days=2) > label_cutoff:
        raise ValueError('All D+2 labels must be known by label_end')
    history = history_frame.loc[history_frame.date.le(label_cutoff)].copy().reset_index(drop=True)
    # Verify chronological continuity before deriving the D+2 label by shift.
    scaler = fit_scaler(history, cutoff=cutoff)
    build_windows(history, scaler, 1)
    grouped = history.groupby('object_id', observed=True, sort=False)
    future_alarm, future_date = grouped.alarm_today.shift(-2), grouped.date.shift(-2)
    mask = history.date.le(cutoff)
    expected_date = history.loc[mask, 'date'] + pd.Timedelta(days=2)
    if not future_date.loc[mask].eq(expected_date).all() or future_alarm.loc[mask].isna().any():
        raise ValueError('Refit requires complete D+2 label coverage for every training row')
    if TARGET in history and not history.loc[mask, TARGET].eq(future_alarm.loc[mask]).all():
        raise ValueError('Provided target disagrees with the causal D+2 construction')
    train = history.loc[mask].reset_index(drop=True)
    xtrain = torch.from_numpy(build_windows(train, scaler, checkpoint['window_days']))
    ytrain = torch.tensor(future_alarm.loc[mask].to_numpy(np.float32))
    config = checkpoint['training_configuration']
    torch.manual_seed(checkpoint['seed'])
    np.random.seed(checkpoint['seed'])
    model = ObjectGRU(**checkpoint['architecture'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    loss_fn = nn.BCEWithLogitsLoss()
    steps, started = 0, time.monotonic()
    for _ in range(epochs):
        model.train()
        for batch in torch.randperm(len(ytrain)).split(config['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xtrain[batch]), ytrain[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            steps += 1
    metadata = {'source_checkpoint': str(source_path), 'source_sha256': checksum(source_path),
                'train_feature_end': str(cutoff.date()), 'train_label_end': str(expected_date.max().date()),
                'epochs': epochs, 'optimizer_steps': steps, 'training_rows': len(train),
                'validation_used': False, 'seconds': time.monotonic()-started}
    checkpoint.update(state_dict=model.state_dict(), scaler=scaler, train_end=str(cutoff.date()),
                      best_epoch_full_epoch=True, fixed_epoch_refit=metadata)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)
    metadata.update(model_path=str(destination), model_sha256=checksum(destination))
    dump(destination.with_suffix('.json'), metadata)
    return metadata


def fit_candidate(window_days, frame, scaler, train_idx, tune_idx):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    windows = build_windows(frame, scaler, window_days)
    xtrain = torch.from_numpy(windows[train_idx])
    xtune = torch.from_numpy(windows[tune_idx])
    del windows
    ytrain = torch.tensor(frame.loc[train_idx, TARGET].to_numpy(np.float32))
    ytune = frame.loc[tune_idx, TARGET].to_numpy(np.int8)
    architecture = {'input_size': xtrain.shape[-1], 'hidden_size': CONFIG['hidden_size'],
                    'head_size': CONFIG['head_size'], 'layers': CONFIG['layers']}
    model = ObjectGRU(**architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    loss_fn = nn.BCEWithLogitsLoss()
    history, stale, best_ap, best_epoch, best_state = [], 0, -np.inf, 0, None
    stopping_ap = -np.inf
    started = time.monotonic()
    for epoch in range(1, CONFIG['max_epochs']+1):
        if best_state is not None and time.monotonic()-started >= CONFIG['seconds_per_candidate']:
            break
        epoch_start = time.monotonic()
        model.train()
        total_loss, seen = 0., 0
        for batch in torch.randperm(len(ytrain)).split(CONFIG['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xtrain[batch]), ytrain[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total_loss += float(loss.detach())*len(batch)
            seen += len(batch)
            if time.monotonic()-started >= CONFIG['seconds_per_candidate']:
                break
        if seen != len(ytrain):
            history.append({'epoch': epoch, 'tune_ap': None, 'train_loss': total_loss/seen,
                            'full_epoch': False, 'seconds': time.monotonic()-epoch_start})
            log(f'GRU {window_days}d epoch={epoch} interrupted; ineligible for checkpoint selection')
            break
        score = predict_tensor(model, xtune)
        ap = float(average_precision_score(ytune, score))
        history.append({'epoch': epoch, 'tune_ap': ap, 'train_loss': total_loss/seen,
                        'full_epoch': seen == len(ytrain), 'seconds': time.monotonic()-epoch_start})
        if ap > best_ap:
            best_ap, best_epoch = ap, epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if ap > stopping_ap + CONFIG['minimum_ap_improvement']:
            stopping_ap, stale = ap, 0
        else:
            stale += 1
        log(f'GRU {window_days}d epoch={epoch} tune_AP={ap:.6f} best={best_ap:.6f}')
        if stale >= CONFIG['patience']:
            break
    if best_state is None:
        raise RuntimeError('No completed GRU checkpoint')
    model.load_state_dict(best_state)
    scores = predict_tensor(model, xtune)
    checkpoint = {'architecture': architecture, 'state_dict': best_state,
                  'scaler': scaler, 'window_days': window_days, 'seed': SEED,
                  'target': TARGET, 'train_end': str(TRAIN_END.date()),
                  'best_epoch': best_epoch, 'best_epoch_full_epoch': True, 'training_configuration': CONFIG}
    path = OUT / 'models' / f'gru_w{window_days}.pt'
    torch.save(checkpoint, path)
    restored = load_predictor(path)(frame, tune_idx)
    difference = float(np.max(np.abs(restored-scores)))
    if difference > 1e-7:
        raise AssertionError(f'Saved GRU inference differs: {difference}')
    metadata = {'name': f'gru_w{window_days}', 'window_days': window_days,
                'best_epoch': best_epoch, 'epochs_run': len(history),
                'tune_ap': float(average_precision_score(ytune, scores)),
                'seconds': time.monotonic()-started, 'model_sha256': checksum(path),
                'model_path': str(path.relative_to(ROOT)), 'reload_max_difference': difference,
                'parameter_count': sum(p.numel() for p in model.parameters()), 'history': history}
    return scores, metadata


def check_causality(frame, scaler, length):
    cut = pd.Timestamp('2025-10-17')
    before = build_windows(frame, scaler, length)
    changed = frame.copy()
    future = changed.date.gt(cut)
    changed.loc[future, COUNT_COLUMNS] = changed.loc[future, COUNT_COLUMNS]*1000+1000
    changed.loc[future, BINARY_COLUMNS] = 1-changed.loc[future, BINARY_COLUMNS]
    after = build_windows(changed, scaler, length)
    past = frame.date.le(cut).to_numpy()
    delta = float(np.max(np.abs(before[past]-after[past])))
    if delta != 0:
        raise AssertionError('Future mutation changed earlier sequence')
    changed_scaler = fit_scaler(changed)
    if scaler != changed_scaler:
        raise AssertionError('Future mutation changed train scaler')
    return {'window_days': length, 'future_changed_after': str(cut.date()),
            'earlier_windows_max_difference': delta, 'train_scaler_unchanged': True,
            'sequence_endpoint_checked': True, 'daily_contiguity_checked': True}


def main():
    started = time.monotonic()
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    if not (OUT/'protocol.md').exists():
        raise RuntimeError('Write protocol before running candidates')
    (OUT/'models').mkdir(parents=True, exist_ok=True)
    (OUT/'predictions').mkdir(exist_ok=True)
    raw, future_date, _ = probe.prepare(end_date='2025-11-30')
    keep = raw.date.le(LAST_NEEDED_DAY)
    frame = raw.loc[keep].reset_index(drop=True)
    label_date = future_date.loc[keep].reset_index(drop=True)
    del raw, future_date
    train_idx = frame.index[frame.date.le(TRAIN_END)].to_numpy()
    tune_idx = frame.index[frame.date.between(TUNE_START, TUNE_END)].to_numpy()
    assert label_date.loc[train_idx].max() <= pd.Timestamp('2025-09-30')
    assert label_date.loc[tune_idx].max() <= LAST_NEEDED_DAY
    assert frame.loc[np.r_[train_idx, tune_idx], TARGET].notna().all()
    assert frame.object_id.nunique() == 78
    assert label_date.loc[tune_idx].eq(frame.loc[tune_idx, 'date']+pd.Timedelta(days=2)).all()
    scaler = fit_scaler(frame)
    yt = frame.loc[train_idx, TARGET].to_numpy(np.int8)
    yv = frame.loc[tune_idx, TARGET].to_numpy(np.int8)
    frequency = frame.loc[train_idx].groupby('object_id', observed=True)[TARGET].mean()
    scores = {'always_positive': np.ones(len(tune_idx)),
              'persistence_alarm_today': frame.loc[tune_idx, 'alarm_today'].to_numpy(),
              'frozen_object_frequency': frame.loc[tune_idx, 'object_id'].map(frequency).astype(float).to_numpy()}
    prediction = frame.loc[tune_idx, ['object_id', 'date', TARGET, 'alarm_today']].copy().reset_index(drop=True)
    previous_path = probe.CACHE / 'object_risk_tuning_predictions.parquet'
    comparator_status = 'unavailable'
    if previous_path.exists():
        previous = pd.read_parquet(previous_path, filters=[('date','>=',TUNE_START),('date','<=',TUNE_END)])
        joined = prediction[['object_id','date',TARGET]].merge(previous[['object_id','date',TARGET,'lightgbm_object']],
                    on=['object_id','date',TARGET], how='left', validate='one_to_one', sort=False)
        if len(previous) != len(prediction) or joined.lightgbm_object.isna().any():
            raise ValueError('Stored object LightGBM tuning identities/labels differ')
        scores['lightgbm_object'] = joined.lightgbm_object.to_numpy()
        comparator_status = 'tuning keys and targets match exactly'
    candidates, causal_checks = [], []
    log(f'Train={len(train_idx)}, tune={len(tune_idx)}, objects=78; CPU threads=3')
    for window in [28, 56]:
        causal_checks.append(check_causality(frame, scaler, window))
        score, metadata = fit_candidate(window, frame, scaler, train_idx, tune_idx)
        scores[metadata['name']] = score
        candidates.append(metadata)
        dump(OUT/'training_progress.json', candidates)
    winner = max(candidates, key=lambda item: item['tune_ap'])
    rows, thresholds, feasibility = [], {}, {}
    quiet = frame.loc[tune_idx, 'alarm_today'].eq(0).to_numpy()
    days = frame.loc[tune_idx, 'date'].nunique()
    for name, score in scores.items():
        threshold, facts = probe.threshold_and_feasibility(yv, score)
        if name in ['always_positive', 'persistence_alarm_today']:
            threshold = .5
        thresholds[name], feasibility[name] = threshold, facts
        rows.append(probe.metrics(name, 'all_object_days', yv, score, threshold, days))
        rows.append(probe.metrics(name, 'no_alarm_on_feature_day', yv[quiet], score[quiet], threshold, days))
        prediction[name] = score
    table = pd.DataFrame(rows)
    table.to_csv(OUT/'tuning_metrics.csv', index=False)
    prediction['selected_score'] = scores[winner['name']]
    prediction.to_parquet(OUT/'predictions/tune.parquet', index=False)
    dump(OUT/'causality_checks.json', causal_checks)
    dump(OUT/'feasibility.json', feasibility)
    manifest = {'selected_model': winner['name'], 'selection_criterion': 'AP on all Oct-Nov tuning rows',
                'model_path': winner['model_path'], 'model_sha256': winner['model_sha256'],
                'input_features': [*BINARY_COLUMNS, *COUNT_COLUMNS, 'history_present'], 'scaler': scaler,
                'target': TARGET, 'minimum_lead_hours': 24, 'sequence_endpoint': 'feature day D',
                'train_feature_end': str(TRAIN_END.date()), 'train_label_end': '2025-09-30',
                'tune_feature_start': str(TUNE_START.date()), 'tune_feature_end': str(TUNE_END.date()),
                'tune_label_end': '2025-11-30', 'preparation_materialized_through': '2025-11-30',
                'retained_history_end': str(LAST_NEEDED_DAY.date()),
                'december_used_for_fit_or_selection': False, 'test_2026_read': False,
                'calibration_or_refit_performed': False, 'objects': 78,
                'training_rows': len(train_idx), 'tuning_rows': len(tune_idx),
                'training_prevalence': float(yt.mean()), 'tuning_prevalence': float(yv.mean()),
                'tuning_thresholds_research_only': thresholds, 'candidates': candidates,
                'lightgbm_comparator_status': comparator_status,
                'input_sha256': {str(p.relative_to(ROOT)): checksum(p) for p in [probe.CACHE/'daily_2025.parquet',
                    ROOT/'data/raw/справочник_каналов_датчиков.csv',ROOT/'data/raw/справочник_объектов_диспетчер.csv']},
                'code_sha256': checksum(Path(__file__)), 'probe_code_sha256': checksum(Path(probe.__file__)),
                'protocol_sha256': checksum(OUT/'protocol.md'),
                'versions': {name: importlib.metadata.version(name) for name in ['torch','numpy','pandas','scikit-learn']},
                'elapsed_seconds': time.monotonic()-started}
    dump(OUT/'configuration.json', manifest)
    all_rows = table[table.cohort.eq('all_object_days')]
    metric_table = ['| Модель | AP | Precision | Recall | F1 |', '|---|---:|---:|---:|---:|']
    for row in all_rows.itertuples():
        metric_table.append(f'| {row.model} | {row.average_precision:.6f} | {row.precision:.6f} | {row.recall:.6f} | {row.f1:.6f} |')
    report = ['# Компактная временная нейросеть: исследование на объектной цели', '',
              'Train до 28.09.2025; tune 01.10–28.11.2025; цель — тревожная запись на D+2. '
              'Подготовка ограничена 30.11.2025; декабрь и 2026 не использовались для выбора. '
              'Калибровка и refit не выполнялись.', '',
              '\n'.join(metric_table), '',
              f'Выбрана {winner["name"]}, эпоха {winner["best_epoch"]}; '
              f'AP = {winner["tune_ap"]:.6f}. Результат относится только к tune.', '',
              'Проверки окон: будущие изменения не влияют на прошлые окна, scaler обучен только на train. '
              'Сохранённые модели воспроизводят все прогнозы tune с отклонением не более 1e-7.', '',
              'Порог max-F1 выбран на том же tune и не является независимой финальной оценкой. '
              'День без тревоги в момент выдачи не исключает начала события в промежуточный D+1. '
              'Цель описывает зарегистрированную тревогу, не подтверждённую физическую аварию.']
    (OUT/'report.md').write_text('\n'.join(report)+'\n')
    print(all_rows.to_string(index=False), flush=True)
    log(f'Selected {winner["name"]}; stopped before refit/calibration/test')


if __name__ == '__main__':
    main()
