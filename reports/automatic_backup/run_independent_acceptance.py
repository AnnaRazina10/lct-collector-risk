#!/usr/bin/env python3
"""Black-box scheduler acceptance on a disposable DB extracted from local v2 ZIP.

Never opens a working dispatcher database. All process IDs belong to this script.
Receipts/logs persist under reports/automatic_backup; temporary DBs are removed.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient
from api import main, forecast_store

OUT = ROOT / 'reports/automatic_backup'
SCHEDULER = ROOT / 'scripts/backup_scheduler.py'
BACKUP = ROOT / 'scripts/database_backup.py'
ZIP = ROOT / 'artifacts/submission/bundles/collector-risk-review-v2.zip'
CHECKS, EVENTS, PROCESSES, LOGS = [], [], [], []
STATUS_READS = 0
STARTED = time.monotonic()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def event(name, **fields):
    EVENTS.append({'event': name, 'elapsed_seconds': time.monotonic()-STARTED, **fields})
    print(json.dumps(EVENTS[-1], ensure_ascii=False), flush=True)


def check(name, condition, detail=None):
    if not condition:
        raise AssertionError(name)
    CHECKS.append({'name': name, 'passed': True, 'detail': detail})


def read_status(folder):
    global STATUS_READS
    path = folder/'status.json'
    if not path.exists():
        return None
    value = json.loads(path.read_text())  # Any partial JSON is a failed acceptance.
    STATUS_READS += 1
    return value


def wait_status(folder, predicate, *, timeout=10):
    until = time.monotonic()+timeout
    latest = None
    while time.monotonic() < until:
        latest = read_status(folder)
        if latest is not None and predicate(latest):
            return latest
        time.sleep(.01)
    raise AssertionError('Status condition timed out: '+json.dumps(latest, ensure_ascii=False))


def launch(label, source, folder, *, interval=.2, max_runs=None):
    log = (OUT/(label+'.log')).open('w')
    LOGS.append(log)
    args = [sys.executable, str(SCHEDULER), '--source', str(source), '--output-dir', str(folder),
            '--interval', str(interval), '--timeout', '8']
    if max_runs is not None:
        args.extend(['--max-runs', str(max_runs)])
    process = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    PROCESSES.append(process)
    event('process_started', label=label, pid=process.pid, interval=interval, max_runs=max_runs)
    return process


def stop(process, label):
    before = time.monotonic()
    process.send_signal(signal.SIGTERM)
    code = process.wait(timeout=12)
    duration = time.monotonic()-before
    event('process_stopped', label=label, pid=process.pid, returncode=code, seconds=duration)
    check(label+'_orderly_sigterm_exit', code == 143)
    return duration


def wait_exit(process, label, folder, *, timeout=12):
    until = time.monotonic()+timeout
    while process.poll() is None and time.monotonic() < until:
        read_status(folder)
        time.sleep(.01)
    if process.poll() is None:
        raise TimeoutError(label)
    event('process_completed', label=label, pid=process.pid, returncode=process.returncode)
    return process.returncode


def cli(args, label):
    before = time.monotonic()
    completed = subprocess.run([sys.executable, str(BACKUP), *args], cwd=ROOT,
                               capture_output=True, text=True, timeout=15)
    (OUT/(label+'.log')).write_text(completed.stdout+completed.stderr)
    check(label+'_exit0', completed.returncode == 0, completed.stderr)
    result = json.loads(completed.stdout)
    check(label+'_passed', result.get('passed') is True)
    event(label, seconds=time.monotonic()-before)
    return result


def main_review():
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in [SCHEDULER, BACKUP, ROOT/'api/main.py', ROOT/'src/serving/local_recommendations.py', Path(__file__)]}
    zip_sha = sha(ZIP)
    statuses = {}
    with tempfile.TemporaryDirectory(prefix='collector-scheduler-independent-') as temporary:
        work = Path(temporary)
        source, folder = work/'source.sqlite3', work/'backups'
        with ZipFile(ZIP) as archive:
            manifest = json.loads(archive.read('collector-risk-review/bundle_manifest.json'))
            contents = archive.read('collector-risk-review/data/app/forecast_seed.sqlite3')
        check('zip_seed_sha_matches_manifest', hashlib.sha256(contents).hexdigest() == manifest['files']['data/app/forecast_seed.sqlite3']['sha256'])
        source.write_bytes(contents)
        payloads = {m['run_id']: forecast_store.load_run(source, m['run_id']) for m in forecast_store.list_runs(source, limit=500)}
        check('six_releases_468_cards', len(payloads)==6 and sum(len(p['cards']) for p in payloads.values())==468)
        selected = next(p for p in payloads.values() if p.get('target_kind')=='registered_episode_start_g1')
        card = selected['cards'][0]
        with patch.object(main, 'DB', source), TestClient(main.app) as client:
            response = client.post('/api/feedback', json={'risk_id': card['id'], 'entity_mode': 'object',
                'run_id': selected['run_id'], 'decision': 'inspect', 'reason': 'journal_verified',
                'operator': 'ТЕСТ РЕЗЕРВНОГО КОПИРОВАНИЯ',
                'note': 'Искусственная запись только в временной копии из v2 ZIP.'})
            check('server_test_feedback_created', response.status_code==201)
            feedback = response.json()
            check('server_recommendation_present', feedback['risk_snapshot']['recommendation']['status']=='ok')
        source_sha = sha(source)
        # A real exclusive SQLite lock makes the first backup take longer than its interval.
        lock = sqlite3.connect(source)
        lock.execute('PRAGMA journal_mode=DELETE')
        lock.execute('BEGIN EXCLUSIVE')
        first = launch('automatic_three', source, folder, interval=.2, max_runs=3)
        wait_status(folder, lambda s: s.get('pid')==first.pid)
        competitor = launch('competitor', source, folder, interval=.2, max_runs=1)
        competitor_code = wait_exit(competitor, 'competitor', folder, timeout=3)
        check('concurrent_scheduler_refused_exit3', competitor_code == 3)
        time.sleep(.65)
        lock.rollback(); lock.close()
        event('sqlite_exclusive_lock_released')
        check('automatic_three_exit0', wait_exit(first, 'automatic_three', folder)==0)
        three = read_status(folder); statuses['after_three'] = three
        check('three_successful_attempts', three['session_attempts']==3 and three['counters']['successes']==3 and three['counters']['failures']==0)
        bundles = sorted(p.parent for p in folder.glob('*/manifest.json'))
        check('three_distinct_published_bundles', len(bundles)==3)
        check('real_long_backup_skips_missed_slots', three['skipped_slots']>=3, three['skipped_slots'])
        check('bounded_session_stopped', three['state']=='stopped' and three['stop_reason']=='max_runs')
        for i, bundle in enumerate(bundles):
            cli(['verify','--bundle',str(bundle),'--timeout','8'], f'verify_automatic_{i+1}')
        # Failure and recovery occur in one live process, without replacing its status file.
        second = launch('failure_recovery', source, folder, interval=.4)
        recovered_start = wait_status(folder, lambda s:s.get('pid')==second.pid and s['counters']['successes']>=4)
        last_good = recovered_start['last_success']
        successes_before = recovered_start['counters']['successes']
        hidden = work/'source.temporarily-unavailable.sqlite3'
        source.rename(hidden)
        event('source_hidden_for_failure')
        failed = wait_status(folder, lambda s:s['counters']['failures']>=1)
        statuses['source_failure'] = failed
        check('failure_preserves_last_success', failed['last_success']==last_good)
        check('failure_keeps_successful_bundle', Path(last_good['destination']).is_dir())
        hidden.rename(source)
        event('source_restored_after_failure')
        recovered = wait_status(folder, lambda s:s['counters']['successes']>successes_before)
        statuses['recovered'] = recovered
        check('recovery_updates_last_success', recovered['last_success']['destination'] != last_good['destination'])
        check('recovery_retains_historical_error', recovered['last_error']==failed['last_error'])
        stop_seconds = stop(second, 'failure_recovery')
        stopped = read_status(folder); statuses['after_sigterm'] = stopped
        check('stopped_state_after_signal', stopped['state']=='stopped' and stopped['stop_reason']=='signal:SIGTERM')
        # Restart while source is absent: last successful bundle must survive the new session.
        source.rename(hidden)
        third = launch('restart_missing_source', source, folder, interval=.2, max_runs=1)
        third_code = wait_exit(third, 'restart_missing_source', folder)
        check('bounded_missing_source_exit1', third_code==1)
        restart_failed = read_status(folder); statuses['restart_missing_source'] = restart_failed
        check('restart_failure_retains_success', restart_failed['last_success']==stopped['last_success'])
        check('restart_failure_count_incremented', restart_failed['counters']['failures']==stopped['counters']['failures']+1)
        hidden.rename(source)
        # Signal an actual in-flight backup while its SQLite source is locked.
        active_lock = sqlite3.connect(source); active_lock.execute('BEGIN EXCLUSIVE')
        fourth = launch('inflight_sigterm', source, folder, interval=.2, max_runs=3)
        active = wait_status(folder, lambda s:s.get('pid')==fourth.pid and s['last_attempt']['status']=='in_progress')
        signal_started = time.monotonic(); fourth.send_signal(signal.SIGTERM)
        time.sleep(.08)
        active_lock.rollback();active_lock.close()
        signal_code = wait_exit(fourth, 'inflight_sigterm', folder)
        signal_seconds = time.monotonic()-signal_started
        check('inflight_signal_exit143', signal_code==143)
        after_active = read_status(folder);statuses['inflight_sigterm'] = after_active
        check('inflight_backup_finished_before_stop', after_active['session_attempts']==1 and after_active['session_successes']==1 and after_active['last_attempt']['status']=='succeeded' and after_active['stop_reason']=='signal:SIGTERM')
        check('inflight_signal_did_not_start_more_backups', after_active['counters']['attempts']==restart_failed['counters']['attempts']+1)
        fifth = launch('restart_recovered', source, folder, interval=.2, max_runs=1)
        check('restart_recovered_exit0', wait_exit(fifth, 'restart_recovered', folder)==0)
        final_status = read_status(folder); statuses['restart_recovered'] = final_status
        check('restart_success_count_incremented', final_status['counters']['successes']==after_active['counters']['successes']+1)
        check('restart_preserves_cumulative_attempts', final_status['counters']['attempts']==after_active['counters']['attempts']+1)
        check('source_db_bytes_unchanged_by_scheduler', sha(source)==source_sha)
        latest = Path(final_status['last_success']['destination'])
        restored = work/'restored-new.sqlite3'
        restore_receipt = cli(['restore','--bundle',str(latest),'--destination',str(restored),'--timeout','8'], 'restore_latest')
        restored_payloads = {m['run_id']:forecast_store.load_run(restored,m['run_id']) for m in forecast_store.list_runs(restored,limit=500)}
        check('restore_exact_six_forecast_payloads', restored_payloads == payloads)
        with patch.object(main,'DB',restored), TestClient(main.app) as client:
            journal = client.get('/api/journal?mode=object').json()['entries']
        restored_feedback = next(x for x in journal if x['id']==feedback['id'])
        check('restore_exact_server_recommendation_snapshot', restored_feedback['risk_snapshot']==feedback['risk_snapshot'])
        check('restore_exact_dispatcher_test_fields', all(restored_feedback[key]==feedback[key] for key in ['risk_id','entity_mode','decision','reason','operator','note','created_at']))
        check('restored_snapshot_not_regenerated', restored_feedback['risk_snapshot']['recommendation']['generated_at']==feedback['risk_snapshot']['recommendation']['generated_at'])
        snapshot_hash = hashlib.sha256(json.dumps(feedback['risk_snapshot'],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        check('input_zip_unchanged', sha(ZIP)==zip_sha)
        check('scheduler_sources_unchanged_during_acceptance', all(sha(ROOT/path)==value for path,value in source_hashes.items()))
        receipt = {'status':'passed','scope':'Local periodic backup, process exclusion, failure/recovery, orderly stop/restart and exact restore only; not complete section11, HA, retention, off-host storage or deployment validation.',
            'source_zip':str(ZIP.relative_to(ROOT)),'source_zip_sha256':zip_sha,
            'source_seed_sha256':manifest['files']['data/app/forecast_seed.sqlite3']['sha256'],
            'source_sha256':source_hashes,'checks':CHECKS,'checks_count':len(CHECKS),'status_reads_without_partial_json':STATUS_READS,
            'events':EVENTS,'status_snapshots':statuses,'sigterm_exit_seconds':stop_seconds,'inflight_sigterm_exit_seconds':signal_seconds,
            'restart_missing_source_exit_code':third_code,'restore_elapsed_seconds':restore_receipt['elapsed_seconds'],
            'restored_releases':len(restored_payloads),'restored_cards':sum(len(p['cards']) for p in restored_payloads.values()),
            'restored_test_recommendation_snapshot_sha256':snapshot_hash,
            'restored_payload_sha256':{key:hashlib.sha256(json.dumps(p,sort_keys=True,ensure_ascii=False).encode()).hexdigest() for key,p in restored_payloads.items()},
            'total_elapsed_seconds':time.monotonic()-STARTED,'working_databases_opened':False,
            'temporary_databases_removed_after_check':True}
    (OUT/'independent_review.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    event('acceptance_complete', checks=len(CHECKS), seconds=receipt['total_elapsed_seconds'])
    return receipt


if __name__ == '__main__':
    try:
        main_review()
    finally:
        for process in PROCESSES:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait()
        for log in LOGS:
            log.close()
