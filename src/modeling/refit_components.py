"""Refit a selected GBDT graph through Nov 29 using already selected tree counts.

No tuning, evaluation set, early stopping, calibration labels or test labels are
used here. Original tuning artifacts are retained. TabM stays frozen.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import inspect
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

TARGET = 'target_alarm_next_24h'
CUTOFF = pd.Timestamp('2025-11-29')
CAP = 1_000_000
SEED = 84


def _sample(mart, idx, cap=CAP):
    """Same seed, inclusion scheme and inverse-probability weights as v2.sample."""
    y = mart.loc[idx, TARGET].to_numpy()
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError('Refit training labels must contain both binary classes')
    positive, negative = idx[y == 1], idx[y == 0]
    rng = np.random.default_rng(SEED)
    p = rng.choice(positive, min(len(positive), cap // 3), replace=False)
    n = rng.choice(negative, min(len(negative), cap - len(p)), replace=False)
    sampled = np.r_[p, n]
    rng.shuffle(sampled)
    sampled_y = mart.loc[sampled, TARGET].to_numpy()
    weight = np.where(sampled_y == 1, len(positive)/len(p), len(negative)/len(n))
    return sampled, weight / weight.mean()


def _frame(mart, idx, features, cats, catboost):
    if {TARGET, 'date', 'split', 'channel_id'} & set(features):
        raise ValueError('Label, split, date or raw row identifier in feature list')
    frame = mart.loc[idx, features].copy()
    for col in cats:
        if col not in frame:
            raise ValueError(f'Categorical feature {col} absent from model features')
        if catboost:
            frame[col] = frame[col].astype(str)
        else:
            # Build pandas category metadata from this refit sample only. LightGBM
            # saves it with the booster and aligns prediction categories itself.
            frame[col] = frame[col].astype('category').cat.remove_unused_categories()
    if catboost:
        for col in frame.select_dtypes(include='number'):
            frame[col] = frame[col].replace([np.inf, -np.inf], np.nan).fillna(-999999)
    return frame


def _cat_params(source):
    """Freeze public resolved parameters, remove validation-only training state."""
    all_params = source.get_all_params()
    accepted = set(inspect.signature(CatBoostClassifier.__init__).parameters) - {'self'}
    params = {k: v for k, v in all_params.items() if k in accepted}
    # Preserve explicit constructor parameters absent from get_all_params, too.
    params.update({k: v for k, v in source.get_params().items() if k in accepted})
    for key in ['od_type', 'od_wait', 'od_pval', 'early_stopping_rounds',
                'verbose', 'silent', 'logging_level', 'best_model_min_trees',
                'n_estimators', 'num_trees', 'num_boost_round', 'eval_fraction']:
        params.pop(key, None)
    if params.get('grow_policy', 'SymmetricTree') != 'Lossguide':
        params.pop('max_leaves', None)
        params.pop('num_leaves', None)
    if params.get('grow_policy', 'SymmetricTree') == 'SymmetricTree':
        params.pop('min_data_in_leaf', None)
        params.pop('min_child_samples', None)
    params.update(iterations=int(source.tree_count_), thread_count=5,
                  use_best_model=False, allow_writing_files=False, task_type='CPU')
    return params, sorted(set(all_params) - accepted)


def _lgb_params(source):
    params = copy.deepcopy(source.params)
    for key in ['num_iterations', 'num_iteration', 'num_tree', 'num_trees',
                'num_round', 'num_rounds', 'nrounds', 'num_boost_round',
                'n_estimators', 'max_iter', 'early_stopping_round',
                'early_stopping_rounds', 'early_stopping', 'n_iter_no_change',
                'early_stopping_min_delta', 'categorical_feature',
                'categorical_column', 'cat_feature', 'cat_column']:
        params.pop(key, None)
    params['num_threads'] = 5
    return params


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def refit_components(winner, configs, mart, out_dir):
    """Return an independent config graph pointing only selected GBDTs to refits.

    ``out_dir`` is the experiment report directory containing ``models/``.
    Each ordinary component gets ``artifact_name='refit_NAME'``. Expert maps
    point directly to their refit artifact names, matching v2's expert loader.
    Call only after checking old tuning predictions; calibrate returned configs
    subsequently on December. This function never reads December labels.
    """
    result = copy.deepcopy(configs)
    if winner not in result:
        raise KeyError(f'Unknown selected model: {winner}')
    if not mart.index.is_unique:
        raise ValueError('Refit matrix must have unique row indexes')
    output = Path(out_dir)
    model_dir = output / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    eligible = mart.index[pd.to_datetime(mart['date']) <= CUTOFF]
    if not len(eligible):
        raise ValueError('No training observations before refit cutoff')
    sampled, shared_weight = _sample(mart, eligible)
    manifest = {'selected_model': winner, 'cutoff_feature_date': str(CUTOFF.date()),
                'last_allowed_target_day': '2025-11-30', 'sample_cap': CAP,
                'sampling_seed': SEED, 'eligible_rows': len(eligible),
                'sampled_rows': len(sampled), 'validation_used': False,
                'early_stopping_used': False, 'future_labels_accessed': False,
                'components': {}, 'status': 'in_progress'}
    manifest_path = output / 'refit_manifest.json'

    def save_manifest():
        temporary = manifest_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
        temporary.replace(manifest_path)

    def train_component(name, cfg, indices, weight, system=None):
        if cfg.get('external_v1'):
            raise ValueError('v1_control used December for tuning and is not eligible for refit')
        source_name = cfg.get('artifact_name', name)
        if source_name.startswith('refit_'):
            raise ValueError(f'Expected original tuning artifact, got {source_name}')
        if Path(source_name).name != source_name or Path(name).name != name:
            raise ValueError('Model artifact names must be basenames')
        destination_name = f'refit_{name}'
        kind = cfg['kind']
        suffix = '.cbm' if kind == 'catboost' else '.txt'
        source_path = model_dir / f'{source_name}{suffix}'
        destination = model_dir / f'{destination_name}{suffix}'
        features, cats = cfg['features'], cfg['cats']
        x = _frame(mart, indices, features, cats, kind == 'catboost')
        y = mart.loc[indices, TARGET].to_numpy()
        if len(np.unique(y)) != 2:
            raise ValueError(f'Single-class refit sample for {name}')
        dates = pd.to_datetime(mart.loc[indices, 'date'])
        if dates.max() > CUTOFF:
            raise AssertionError('Refit sample crosses the time cutoff')
        weight = weight.copy()
        recent = bool(cfg.get('recent', False) or cfg.get('recent_half_life_days') or name == 'catboost_recent')
        if recent:
            weight *= np.power(.5, (CUTOFF - dates).dt.days.to_numpy()/90)
            weight /= weight.mean()
        start = time.monotonic()
        if kind == 'catboost':
            source = CatBoostClassifier()
            source.load_model(str(source_path))
            if list(source.feature_names_) != list(features):
                raise ValueError(f'Feature order differs from original model: {name}')
            trees = int(source.tree_count_)
            params, omitted = _cat_params(source)
            model = CatBoostClassifier(**params)
            print(f'Refit {name}: rows={len(y)}, trees={trees}, through={CUTOFF.date()}', flush=True)
            model.fit(x, y, sample_weight=weight, cat_features=cats, verbose=False)
            actual_trees = int(model.tree_count_)
        elif kind == 'lgb':
            source = lgb.Booster(model_file=str(source_path))
            if list(source.feature_name()) != list(features):
                raise ValueError(f'Feature order differs from original model: {name}')
            trees = int(source.current_iteration())
            params, omitted = _lgb_params(source), []
            data = lgb.Dataset(x, label=y, weight=weight, categorical_feature=cats,
                               feature_name=features, free_raw_data=True)
            print(f'Refit {name}: rows={len(y)}, trees={trees}, through={CUTOFF.date()}', flush=True)
            model = lgb.train(params, data, num_boost_round=trees)
            actual_trees = int(model.current_iteration())
            del data
        else:
            raise ValueError(f'Unsupported refit model kind: {kind}')
        if actual_trees != trees:
            raise RuntimeError(f'{name}: fixed {trees} trees requested, got {actual_trees}')
        model.save_model(str(destination))
        identities = mart.loc[indices, ['channel_id', 'date']]
        manifest['components'][name] = {
            'kind': kind, 'system': system, 'source_artifact': str(source_path),
            'source_sha256': _sha(source_path), 'artifact_name': destination_name,
            'artifact_sha256': _sha(destination), 'selected_trees': trees,
            'actual_trees': actual_trees, 'sampled_rows': len(y), 'positives': int(y.sum()),
            'unique_channels': int(identities.channel_id.nunique()),
            'row_identity_sha256': hashlib.sha256(pd.util.hash_pandas_object(identities, index=False).to_numpy().tobytes()).hexdigest(),
            'first_feature_date': str(dates.min().date()), 'last_feature_date': str(dates.max().date()),
            'features': features, 'categorical_features': cats,
            'categorical_levels': {c: list(x[c].cat.categories) for c in cats} if kind == 'lgb' else None,
            'weighting': 'global stratified inverse inclusion probability',
            'recent_half_life_days': 90 if recent else None,
            'recent_anchor': str(CUTOFF.date()) if recent else None,
            'training_params': params, 'catboost_internal_parameters_not_constructor_args': omitted,
            'seconds': time.monotonic()-start}
        save_manifest()
        del source, model, x
        gc.collect()
        return destination_name

    completed, active, specialist_artifacts = set(), set(), {}

    def visit(name):
        if name in completed:
            return
        if name in active:
            raise ValueError(f'Cycle in selected model graph at {name}')
        active.add(name)
        cfg = result[name]
        if cfg['kind'] == 'blend':
            for component in cfg['parts']:
                visit(component)
        elif cfg['kind'] == 'experts':
            visit(cfg['fallback'])
            systems = mart.loc[sampled, 'engineering_system'].astype(str).to_numpy()
            for system, original_name in list(cfg['specialists'].items()):
                signature = (original_name, str(system), tuple(cfg['features']), tuple(cfg['cats']))
                if original_name in specialist_artifacts:
                    old_signature, artifact = specialist_artifacts[original_name]
                    if old_signature != signature:
                        raise ValueError(f'Specialist artifact reused with conflicting context: {original_name}')
                else:
                    mask = systems == str(system)
                    if not mask.any():
                        raise ValueError(f'No refit observations for selected specialist {system}')
                    spec = {'kind': 'lgb', 'features': cfg['features'], 'cats': cfg['cats']}
                    artifact = train_component(original_name, spec, sampled[mask], shared_weight[mask], str(system))
                    specialist_artifacts[original_name] = (signature, artifact)
                cfg['specialists'][system] = artifact
        elif cfg['kind'] == 'tabm':
            manifest['components'][name] = {'kind': 'tabm', 'status': 'frozen', 'train_end': '2025-09-29'}
        elif cfg['kind'] in {'lgb', 'catboost'}:
            cfg['artifact_name'] = train_component(name, cfg, sampled, shared_weight)
            cfg['refit_train_end'] = str(CUTOFF.date())
        else:
            raise ValueError(f'Unknown selected model kind: {cfg["kind"]}')
        active.remove(name)
        completed.add(name)

    save_manifest()
    visit(winner)
    manifest['status'] = 'complete'
    manifest['selected_graph_nodes'] = sorted(completed)
    save_manifest()
    return result
