#!/usr/bin/env python3
"""Fixed-trace loopback comparison, with owned processes and isolated seed copies.

Does not open the user journal. Measures archived serving, never ML inference.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import quote
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api import forecast_store
from scripts.verify_runtime import csv_file, distribution, dump, environment, sha256


def trace_for(payloads, cycles, scenario):
    result = []
    for cycle in range(cycles):
        value = payloads[cycle % len(payloads)]
        card = value['cards'][(cycle // len(payloads)) % len(value['cards'])]
        suffix = '?mode=object&run_id=' + quote(value['run_id'], safe='')
        identity = {'risk_id': card['id'], 'entity_mode': 'object', 'run_id': value['run_id']}
        feedback = {**identity, 'decision': 'monitor', 'reason': 'journal_verified',
                    'operator': 'LOAD TEST ONLY', 'note': f'TEST cycle {cycle:04d}'}
        ticket = {**identity, 'note': f'TEST cycle {cycle:04d}'}
        routes = [('runs', '/api/forecast-runs?mode=object'), ('queue', '/api/risks'+suffix),
                  ('card', '/api/risks/'+quote(card['id'], safe='')+suffix),
                  ('journal', '/api/journal?mode=object'), ('queue', '/api/risks'+suffix),
                  ('feedback', '/api/feedback?mode=object'),
                  ('card', '/api/risks/'+quote(card['id'], safe='')+suffix),
                  ('tickets', '/api/tickets?mode=object'),
                  ('feedback', '/api/feedback?mode=object'), ('tickets', '/api/tickets?mode=object')]
        for position, (kind, path) in enumerate(routes):
            post = scenario == 'mixed' and position >= 8
            result.append({'index': len(result), 'cycle': cycle, 'kind': kind,
                           'method': 'POST' if post else 'GET',
                           'path': path.split('?')[0] if post else path,
                           'body': (feedback if position == 8 else ticket) if post else None,
                           'run_id': value['run_id'], 'card_id': card['id']})
    return result


def request(client, operation):
    body = json.dumps(operation['body']).encode() if operation.get('body') else None
    client.request(operation['method'], operation['path'], body=body,
                   headers={'Content-Type': 'application/json'} if body else {})
    response = client.getresponse()
    raw = response.read()
    return response.status, len(raw), json.loads(raw)


def validate(op, status, value, payloads, cards, readonly):
    assert status == (201 if op['method'] == 'POST' else 200), status
    if op['method'] == 'POST':
        assert value['risk_id'] == op['card_id'] and value['external_submission'] is False
        assert value['risk_snapshot']['card'] == cards[op['card_id']]
        assert value['risk_snapshot']['recommendation']['status'] == 'ok'
        assert value['risk_snapshot']['recommendation']['card_id'] == op['card_id']
        assert value['risk_snapshot']['recommendation']['run_id'] == op['run_id']
        if op['kind'] == 'tickets':
            assert value['already_exists'] is False
        return
    if op['kind'] == 'runs':
        assert {r['run_id'] for r in value['runs']} == set(payloads)
    elif op['kind'] == 'queue':
        assert value['cards'] == payloads[op['run_id']]['cards']
        assert len(value['recommendations']) == 78
        assert all(x['status'] == 'ok' for x in value['recommendations'].values())
    elif op['kind'] == 'card':
        assert value['card'] == cards[op['card_id']]
        assert value['recommendation']['status'] == 'ok'
    else:
        records = value[{'journal': 'entries', 'feedback': 'feedback', 'tickets': 'tickets'}[op['kind']]]
        if readonly:
            assert records == []
        for record in records:
            assert record['note'].startswith('TEST cycle ')
            assert record['risk_snapshot']['card'] == cards[record['risk_id']]
            assert record['risk_snapshot']['recommendation']['status'] == 'ok'


def one_series(seed, payloads, operations, users, folder):
    folder.mkdir()
    cards = {c['id']: c for p in payloads.values() for c in p['cards']}
    readonly = all(op['method'] == 'GET' for op in operations)
    process = None
    load_before = os.getloadavg()
    with tempfile.TemporaryDirectory(prefix='lct-fixed-load-') as tmp, (folder/'server.log').open('w') as log:
        database = Path(tmp)/'dispatch.sqlite3'
        database.write_bytes(seed)
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0)); listener.listen(128)
            port = listener.getsockname()[1]
            command = [sys.executable, '-m', 'uvicorn', 'api.main:app', '--fd', str(listener.fileno()),
                       '--workers', '1', '--no-access-log', '--log-level', 'warning']
            try:
                process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, 'LCT_TICKET_DB': str(database)},
                                           stdout=log, stderr=subprocess.STDOUT, pass_fds=(listener.fileno(),))
                startup = time.monotonic()
                while True:
                    if process.poll() is not None: raise RuntimeError('Owned API exited during startup')
                    client = HTTPConnection('127.0.0.1', port, timeout=2)
                    try:
                        status, _, health = request(client, {'method':'GET','path':'/api/health'})
                        if status == 200 and health['status'] == 'ok': break
                    except OSError:
                        if time.monotonic()-startup > 20: raise
                        time.sleep(.05)
                    finally: client.close()
                # Warm every release and read route; POST only occurs inside the trace.
                client = HTTPConnection('127.0.0.1', port, timeout=30)
                for op in operations[:60]:
                    if op['method'] == 'GET':
                        status, _, value = request(client, op)
                        validate(op, status, value, payloads, cards, True)
                client.close()
                barrier = threading.Barrier(users+1)
                guard = threading.Lock()
                state = {'next': 0, 'inflight': 0, 'maximum': 0, 'started': 0.0}

                def worker(user):
                    rows = []
                    client = HTTPConnection('127.0.0.1', port, timeout=30)
                    barrier.wait()
                    try:
                        while True:
                            with guard:
                                index = state['next']
                                if index >= len(operations): break
                                state['next'] += 1
                                state['inflight'] += 1
                                state['maximum'] = max(state['maximum'], state['inflight'])
                            op = operations[index]
                            began = time.perf_counter()
                            status, size, error, latency, record_id = None, 0, '', None, ''
                            try:
                                status, size, value = request(client, op)
                                latency = (time.perf_counter()-began)*1000
                                validate(op, status, value, payloads, cards, readonly)
                                if op['method']=='POST': record_id=value['id']
                            except Exception as exc:
                                error = f'{type(exc).__name__}: {exc}'
                                client.close()
                            finally:
                                with guard: state['inflight'] -= 1
                            rows.append({'index':index,'user':user,'cycle':op['cycle'],
                                         'route':op['method']+' '+op['kind'],
                                         'start_offset_seconds':began-state['started'],
                                         'latency_ms':latency if latency is not None else (time.perf_counter()-began)*1000,
                                         'status':status,'response_bytes':size,'record_id':record_id,'error':error})
                    finally: client.close()
                    return rows

                with ThreadPoolExecutor(max_workers=users) as pool:
                    futures = [pool.submit(worker, i) for i in range(users)]
                    state['started'] = time.perf_counter(); barrier.wait()
                    rows = [r for future in futures for r in future.result()]
                    elapsed = time.perf_counter()-state['started']
                csv_file(folder/'requests.csv', sorted(rows, key=lambda r:r['index']))
                # Verify all stored writes directly, avoiding the API's 200-row history limit.
                with sqlite3.connect(database) as conn:
                    assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
                    actual = {}
                    for table, kind in [('feedback','feedback'),('tickets','tickets')]:
                        entries = conn.execute(f'SELECT risk_id,note,risk_snapshot FROM {table}').fetchall()
                        expected = [op for op in operations if op['method']=='POST' and op['kind']==kind]
                        assert len(entries)==len(expected), (table,len(entries),len(expected))
                        assert sorted((r[0],r[1]) for r in entries)==sorted((o['card_id'],o['body']['note']) for o in expected)
                        stored_ids={r[0] for r in conn.execute(f'SELECT id FROM {table}')}
                        ack_ids=[r['record_id'] for r in rows if r['route']=='POST '+kind and not r['error']]
                        assert len(ack_ids)==len(set(ack_ids)) and stored_ids==set(ack_ids)
                        for identity, _, raw in entries:
                            snap=json.loads(raw)
                            assert snap['card']==cards[identity] and snap['recommendation']['status']=='ok'
                        actual[table]=len(entries)
                    journal_mode=conn.execute('PRAGMA journal_mode').fetchone()[0]
                assert all(forecast_store.load_run(database,key)==value for key,value in payloads.items())
                errors=[r for r in rows if r['error']]
                result={'users':users,'requests':len(rows),'errors':len(errors),'wall_seconds':elapsed,
                        'requests_per_second':len(rows)/elapsed,'max_client_inflight':state['maximum'],
                        'latency':distribution([r['latency_ms'] for r in rows]),
                        'by_route':{route:distribution([r['latency_ms'] for r in rows if r['route']==route]) for route in sorted({r['route'] for r in rows})},
                        'response_bytes':sum(r['response_bytes'] for r in rows),'final_rows':actual,
                        'final_snapshot_cards_and_six_payloads_exact':True,
                        'journal_mode':journal_mode,'load_before':list(load_before),'load_after':list(os.getloadavg()),
                        'passed':not errors and len(rows)==len(operations) and state['maximum']==users}
            finally:
                if process is not None:
                    if process.poll() is None: process.terminate()
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
    result['owned_process_reaped']=True
    dump(folder/'summary.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=ROOT/'artifacts/submission/bundles/collector-risk-review-v2.zip')
    parser.add_argument('--cycles',type=int,default=120)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not 1<=args.cycles<=468 or not 1<=args.repeats<=10: parser.error('cycles 1..468, repeats 1..10')
    args.output.mkdir(parents=True,exist_ok=False)
    with ZipFile(args.bundle) as archive:
        manifest=json.loads(archive.read('collector-risk-review/bundle_manifest.json'))
        name='data/app/forecast_seed.sqlite3'
        seed=archive.read('collector-risk-review/'+name)
        import hashlib
        assert hashlib.sha256(seed).hexdigest()==manifest['files'][name]['sha256']
    with tempfile.TemporaryDirectory(prefix='lct-trace-seed-') as tmp:
        source=Path(tmp)/'seed.sqlite3';source.write_bytes(seed)
        payloads={r['run_id']:forecast_store.load_run(source,r['run_id']) for r in forecast_store.list_runs(source)}
    assert len(payloads)==6 and all(len(p['cards'])==78 for p in payloads.values())
    sources=['api/main.py','api/forecast_store.py','src/serving/local_recommendations.py',
             'scripts/compare_runtime.py','scripts/verify_runtime.py']
    hashes={p:sha256(ROOT/p) for p in sources}
    protocol={'cycles':args.cycles,'repeats':args.repeats,'users':[1,20],'scenarios':['read_only','mixed'],
              'source_sha256':hashes,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'bundle_sha256':sha256(args.bundle),'seed_sha256':manifest['files'][name]['sha256'],
              'environment':environment(),
              'scope':'Same fixed operation multiset from identical seed each series; one worker process, persistent connections, no think time. No ML inference. Client/server share CPU. Concurrent completion changes intermediate journal sizes/order; final write counts and all snapshots verified. Two repetitions are descriptive, not SLA or production proof.'}
    dump(args.output/'protocol.json',protocol)
    records=[]
    for scenario in ['read_only','mixed']:
        operations=trace_for(list(payloads.values()),args.cycles,scenario)
        dump(args.output/(scenario+'_trace.json'),operations)
        for repeat in range(args.repeats):
            for users in ([1,20] if repeat%2==0 else [20,1]):
                key=f'{scenario}_r{repeat+1}_u{users}'
                print('Starting '+key,flush=True)
                try:
                    result=one_series(seed,payloads,operations,users,args.output/key)
                except Exception as exc:
                    dump(args.output/'failure.json',{'series':key,'error':f'{type(exc).__name__}: {exc}'})
                    raise
                records.append({'scenario':scenario,'repeat':repeat+1,**result})
                print(json.dumps({k:result[k] for k in ['users','requests','errors','wall_seconds','requests_per_second','passed']}),flush=True)
    unchanged=hashes=={p:sha256(ROOT/p) for p in sources}
    dump(args.output/'summary.json',{'series':records,'source_files_unchanged':unchanged,'passed':unchanged and all(r['passed'] for r in records)})
    return 0 if unchanged and all(r['passed'] for r in records) else 1


if __name__=='__main__':raise SystemExit(main())
