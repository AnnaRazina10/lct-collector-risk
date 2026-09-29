"""Check real replay scores and late-arrival invariance without changing stored runs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.forecast_store import canonical_payload, list_runs, load_run
from src.serving.object_forecast import (
    DailyBatch, assemble_history, features, forecast_times, load_frozen_inputs,
    predict_payload, read_daily,
)


def digest(payload):
    return hashlib.sha256(canonical_payload(payload).encode()).hexdigest()


def main(database, report_dir):
    runs = list_runs(database)
    assert len(runs) == 3
    before = {r['run_id']: digest(load_run(database, r['run_id'])) for r in runs}
    # Comparison uses saved scores only, after generation; no target is loaded here.
    reference = pd.read_parquet(ROOT / 'data/interim/object_forward_predictions.parquet',
                               columns=['object_id', 'date', 'lightgbm_object'])
    comparisons = []
    for run in runs:
        payload = load_run(database, run['run_id'])
        actual = pd.DataFrame(payload['cards'])[['object_id', 'score']]
        expected = reference.loc[reference.date.eq(pd.Timestamp(run['feature_date']))]
        matched = actual.merge(expected, on='object_id', validate='one_to_one')
        assert len(matched) == len(actual) == 78
        difference = float(np.max(np.abs(matched.score - matched.lightgbm_object)))
        assert difference < 1e-7
        comparisons.append({'feature_date': run['feature_date'], 'objects': len(matched),
                            'max_saved_score_difference': difference})
    cfg, model, meta, objects = load_frozen_inputs()
    seed = read_daily('2025-01-01', '2026-06-25')
    batches = [DailyBatch(date, forecast_times(date)['issue_time'], read_daily(date, date))
               for date in ['2026-06-26', '2026-06-27', '2026-06-28']]
    history, provenance = assemble_history(seed, '2026-06-25', batches, '2026-06-26')
    original, _, _ = predict_payload(history, provenance, '2026-06-26', cfg, model, meta, objects)
    original_stored = next(r for r in runs if r['feature_date'] == '2026-06-26')
    assert original['run_id'] == original_stored['run_id']
    assert digest(original) == before[original['run_id']]
    # Hypothetical delayed replacement of a whole daily package, never persisted.
    corrected = batches[0].rows.copy()
    corrected[features.COUNTS] = 0
    corrected['events_count'] = 1
    late = DailyBatch('2026-06-26', '2026-06-27T12:00:00+03:00', corrected)
    # Even invalid contents of a future/unavailable package cannot enter an earlier run.
    poisoned = batches[2].rows.copy()
    poisoned[features.COUNTS] = -1
    future = DailyBatch('2026-06-28', batches[2].available_at, poisoned)
    altered_history, altered_provenance = assemble_history(
        seed, '2026-06-25', [batches[0], batches[1], future, late, batches[0]], '2026-06-26')
    altered, _, _ = predict_payload(altered_history, altered_provenance, '2026-06-26', cfg, model, meta, objects)
    assert digest(altered) == digest(original)
    next_history, next_provenance = assemble_history(seed, '2026-06-25', batches, '2026-06-27')
    next_corrected, next_corrected_provenance = assemble_history(
        seed, '2026-06-25', [*batches, late], '2026-06-27')
    assert next_provenance['input_sha256'] != next_corrected_provenance['input_sha256']
    next_payload, _, _ = predict_payload(next_history, next_provenance, '2026-06-27', cfg, model, meta, objects)
    corrected_payload, _, _ = predict_payload(next_corrected, next_corrected_provenance, '2026-06-27', cfg, model, meta, objects)
    assert next_payload['run_id'] != corrected_payload['run_id']
    def scores(payload):
        return pd.DataFrame(payload['cards']).set_index('object_id').score.sort_index()
    changed = int((scores(next_payload) != scores(corrected_payload)).sum())
    assert changed > 0
    after = {r['run_id']: digest(load_run(database, r['run_id'])) for r in list_runs(database)}
    assert before == after
    result = {'mode': 'historical_replay', 'model_sha256': cfg['model_sha256'],
              'score_reference_comparisons': comparisons,
              'recomputed_payload_identical_to_saved': True,
              'future_and_late_packages_do_not_change_earlier_payload': True,
              'late_correction_changes_later_input_and_run_id': True,
              'later_scores_changed_by_hypothetical_correction': changed,
              'stored_payloads_unchanged': before == after,
              'stored_payload_sha256': before,
              'hypothetical_correction_published': False,
              'limitations': 'Historical availability clock is simulated; no operational event stream or real arrival timestamps.'}
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / 'verification.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,default=ROOT/'data/app/forecast_replay_validated.sqlite3')
    parser.add_argument('--report-dir',type=Path,default=ROOT/'reports/forecast_replay/validated')
    args = parser.parse_args()
    main(args.database, args.report_dir)
