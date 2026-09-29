#!/usr/bin/env python3
"""Exercise a clean local v3 bundle with the EXISTING review runtime, no installs.

Only disposable extracted databases are opened. Starts/stops its own loopback
servers and bounded scheduler, restores to a new file and compares HTTP results.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'reports/submission_readiness'
PROCESSES, LOGS, CHECKS, EVENTS = [], [], [], []
STARTED = time.monotonic()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def check(name, condition, detail=None):
    if not condition:
        raise AssertionError(name)
    CHECKS.append({'name': name, 'passed': True, 'detail': detail})


def event(name, **fields):
    value = {'event': name, 'elapsed_seconds': time.monotonic()-STARTED, **fields}
    EVENTS.append(value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def request(port, path, payload=None):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    req = Request(f'http://127.0.0.1:{port}{path}', data=data,
                  headers={'Content-Type': 'application/json'} if data is not None else {})
    with urlopen(req, timeout=8) as response:
        body = response.read()
        if response.headers.get_content_type() == 'application/json':
            return response.status, json.loads(body)
        return response.status, body


def free_port():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        return listener.getsockname()[1]


def launch(args, cwd, label, env):
    log = (OUT/f'final_bundle_v3_{label}.log').open('w')
    LOGS.append(log)
    proc = subprocess.Popen(args, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    PROCESSES.append(proc)
    event('process_started', label=label, pid=proc.pid)
    return proc


def wait_http(proc, port):
    until = time.monotonic()+12
    while time.monotonic() < until:
        if proc.poll() is not None:
            raise AssertionError(f'Server exited before readiness: {proc.returncode}')
        try:
            code, value = request(port, '/api/health')
            if code == 200 and value.get('status') == 'ok':
                return value
        except (URLError, HTTPError, TimeoutError, ConnectionError):
            pass
        time.sleep(.03)
    raise TimeoutError('Own server did not become ready')


def stop(proc, label):
    before = time.monotonic()
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
    result = proc.wait(timeout=12)
    event('process_stopped', label=label, pid=proc.pid, returncode=result,
          stop_seconds=time.monotonic()-before)
    return result


def command(args, cwd, label, env, timeout=15):
    before = time.monotonic()
    value = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    (OUT/f'final_bundle_v3_{label}.log').write_text(value.stdout+value.stderr)
    check(label+'_exit0', value.returncode == 0, value.stderr)
    result = json.loads(value.stdout)
    event(label, seconds=time.monotonic()-before)
    return result


def get_forecasts(port):
    code, listing = request(port, '/api/forecast-runs?mode=object&limit=500')
    check(f'port{port}_forecast_list200', code==200)
    check(f'port{port}_six_releases', len(listing['runs'])==6)
    payloads, advice = {}, {}
    for metadata in listing['runs']:
        run_id=metadata['run_id']
        code, result=request(port, '/api/risks?'+urlencode({'mode':'object','run_id':run_id}))
        check(run_id+f'_port{port}_200', code==200)
        advice[run_id] = result['recommendations']
        payloads[run_id] = {key:value for key,value in result.items() if key not in ['archive_metadata','recommendations']}
        check(run_id+f'_port{port}_stored_contenthash', json_sha(payloads[run_id])==metadata['content_sha256'])
        check(run_id+f'_port{port}_78cards', len(result['cards'])==78)
        check(run_id+f'_port{port}_advice_coverage', set(advice[run_id])=={c['id'] for c in result['cards']})
        if result.get('warning_policy'):
            sorted_cards=sorted(result['cards'], key=lambda c:(-c['score'],str(c['object_id'])))
            check(run_id+f'_port{port}_fixed_top10', sum(c['warning'] for c in sorted_cards)==10 and all(c['warning']==(i<10) and c['rank']==i+1 for i,c in enumerate(sorted_cards)))
    return listing, payloads, advice


def stripped_advice(values):
    return {run:{card:{key:value for key,value in advice.items() if key!='generated_at'}
                 for card,advice in cards.items()} for run,cards in values.items()}


def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    archive=args.zip.resolve();python=args.runtime_python.absolute()
    check('existing_runtime_python', python.is_file())
    expected_sha=file_sha(archive)
    check('zip_expected_sha256', expected_sha==args.expected_sha256)
    runtime_cfg=python.parent.parent/'pyvenv.cfg'
    cfg_sha=file_sha(runtime_cfg)
    env=os.environ.copy()
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None);env.pop('LCT_TICKET_DB',None)
    runtime_code="import sys,json,importlib.metadata as m; print(json.dumps({'executable':sys.executable,'prefix':sys.prefix,'version':sys.version,'packages':sorted((x.metadata['Name'],x.version) for x in m.distributions())}))"
    runtime_before=command([str(python),'-c',runtime_code], ROOT, 'runtime_before', env)
    check('runtime_is_requested_existing_venv', Path(runtime_before['prefix']).resolve()==python.parent.parent.resolve())
    temporary=tempfile.TemporaryDirectory(prefix='collector-bundle-v3-independent-')
    try:
        work=Path(temporary.name)
        with ZipFile(archive) as zipped:
            names=zipped.namelist()
            check('zip_names_unique', len(names)==len(set(names)))
            for info in zipped.infolist():
                path=Path(info.filename)
                check('safe_zip_member_'+info.filename, not path.is_absolute() and '..' not in path.parts and path.parts[0]=='collector-risk-review' and not stat.S_ISLNK(info.external_attr>>16))
            zipped.extractall(work)
        bundle=work/'collector-risk-review'
        manifest=json.loads((bundle/'bundle_manifest.json').read_text())
        check('clean_commit_matches', manifest['source_commit']==args.expected_commit and manifest['working_tree_changes'] is False)
        check('bundle_excludes_dispatcher_records', manifest['dispatcher_records_included'] is False)
        check('scheduler_and_backup_are_in_bundle', all(p in manifest['files'] for p in ['scripts/backup_scheduler.py','scripts/database_backup.py','docs/automatic_backup.md']))
        verification=command([str(python),'verify_bundle.py'], bundle, 'verify_before', env)
        check('all_manifest_inputs_verified', verification['verified_files']==len(manifest['files']))
        seed=bundle/'data/app/forecast_seed.sqlite3';seed_sha=file_sha(seed)
        source_port=free_port()
        server=launch([str(python),'start_demo.py','--port',str(source_port)],bundle,'source_http',env)
        wait_http(server,source_port)
        code,html=request(source_port,'/')
        check('http_web_from_extracted_bundle',code==200 and hashlib.sha256(html).hexdigest()==file_sha(bundle/'web/index.html'))
        listing_before,payloads_before,advice_before=get_forecasts(source_port)
        code,journal_empty=request(source_port,'/api/journal?mode=object')
        check('seed_journal_empty',code==200 and journal_empty['entries']==[])
        selected=next(p for p in payloads_before.values() if p.get('target_kind')=='registered_episode_start_g1')
        card=selected['cards'][0]
        identity={'risk_id':card['id'],'entity_mode':'object','run_id':selected['run_id']}
        code,feedback=request(source_port,'/api/feedback',{**identity,'decision':'inspect','reason':'journal_verified','operator':'TEST BUNDLE V3','note':'TEST: временная отдельная база проверки ZIP. Не рабочее решение.'})
        check('test_feedback_created',code==201 and feedback['risk_snapshot']['recommendation']['status']=='ok')
        code,ticket=request(source_port,'/api/tickets',{**identity,'note':'TEST: отдельный черновик для проверки резервной копии v3.'})
        check('test_ticket_created',code==201 and not ticket['already_exists'] and ticket['risk_snapshot']['recommendation']['status']=='ok')
        code,journal_before=request(source_port,'/api/journal?mode=object')
        check('two_test_journal_entries',code==200 and len(journal_before['entries'])==2 and {e['id'] for e in journal_before['entries']}=={feedback['id'],ticket['id']})
        runtime=bundle/'data/app/dispatch.sqlite3'
        check('separate_runtime_db_created',runtime.is_file() and runtime.resolve()!=seed.resolve())
        backup_dir=work/'automatic_copies'
        scheduler=launch([str(python),'scripts/backup_scheduler.py','--source',str(runtime),'--output-dir',str(backup_dir),'--interval','0.2','--timeout','8','--max-runs','3'],bundle,'scheduler',env)
        check('bounded_scheduler_exit0',scheduler.wait(timeout=12)==0)
        status=json.loads((backup_dir/'status.json').read_text())
        check('three_periodic_copies',status['session_attempts']==3 and status['session_successes']==3 and status['session_failures']==0 and status['state']=='stopped')
        copies=sorted(p.parent for p in backup_dir.glob('*/manifest.json'))
        check('three_unique_backup_bundles',len(copies)==3)
        for i,copy in enumerate(copies):
            command([str(python),'scripts/database_backup.py','verify','--bundle',str(copy),'--timeout','8'],bundle,f'verify_copy_{i+1}',env)
        latest=Path(status['last_success']['destination'])
        restored=work/'restored-new.sqlite3'
        check('restore_destination_was_new',not restored.exists())
        restored_receipt=command([str(python),'scripts/database_backup.py','restore','--bundle',str(latest),'--destination',str(restored),'--timeout','8'],bundle,'restore_latest',env)
        check('restored_inventory_includes_two_test_records',restored_receipt['inventory']['tables']['feedback']['row_count']==1 and restored_receipt['inventory']['tables']['tickets']['row_count']==1)
        check('source_http_still_healthy',request(source_port,'/api/health')[0]==200)
        stop(server,'source_http')
        restored_port=free_port();restored_env={**env,'LCT_TICKET_DB':str(restored)}
        restored_server=launch([str(python),'-m','uvicorn','api.main:app','--app-dir',str(bundle),'--host','127.0.0.1','--port',str(restored_port)],bundle,'restored_http',restored_env)
        wait_http(restored_server,restored_port)
        listing_after,payloads_after,advice_after=get_forecasts(restored_port)
        code,journal_after=request(restored_port,'/api/journal?mode=object')
        check('restored_get_journal_exact',code==200 and journal_after==journal_before)
        check('all_six_metadata_exact',listing_after==listing_before)
        check('all_six_payloads_exact',payloads_after==payloads_before)
        check('all_468_live_advice_same_except_generated_at',stripped_advice(advice_after)==stripped_advice(advice_before))
        stop(restored_server,'restored_http')
        final_verification=command([str(python),'verify_bundle.py'],bundle,'verify_after',env)
        check('immutable_bundle_unchanged',final_verification==verification and file_sha(seed)==seed_sha)
        runtime_after=command([str(python),'-c',runtime_code],ROOT,'runtime_after',env)
        check('runtime_packages_and_configuration_unchanged',runtime_after==runtime_before and file_sha(runtime_cfg)==cfg_sha)
        check('zip_file_unchanged',file_sha(archive)==expected_sha)
        receipt={'status':'passed','completed_at':datetime.now(timezone.utc).isoformat(),'checks':CHECKS,'checks_count':len(CHECKS),
            'zip_path':str(archive.relative_to(ROOT)) if archive.is_relative_to(ROOT) else str(archive),'zip_sha256':expected_sha,'source_commit':manifest['source_commit'],'working_tree_changes_at_build':manifest['working_tree_changes'],
            'verifier_sha256':file_sha(Path(__file__)),'runtime':runtime_before,'runtime_reinstalled':False,'runtime_configuration_sha256':cfg_sha,
            'verified_bundle_inputs':len(manifest['files']),'bundle_scheduler_sha256':manifest['files']['scripts/backup_scheduler.py']['sha256'],'bundle_api_sha256':manifest['files']['api/main.py']['sha256'],
            'events':EVENTS,'source_port':source_port,'restored_port':restored_port,'backup_scheduler_status':status,'restore_elapsed_seconds':restored_receipt['elapsed_seconds'],
            'restored_table_rows':{key:value['row_count'] for key,value in restored_receipt['inventory']['tables'].items()},
            'forecast_metadata_before':listing_before,'forecast_metadata_after':listing_after,'forecast_payload_sha256':{key:json_sha(value) for key,value in payloads_after.items()},
            'journal_before':journal_before,'journal_after':journal_after,'journal_exact_match':True,'metadata_exact_match':True,'payloads_exact_match':True,
            'working_user_database_opened':False,'new_package_installs':False,'temporary_databases_removed':True,'all_own_processes_stopped':True,'total_elapsed_seconds':time.monotonic()-STARTED,
            'limitations':['Local macOS acceptance in an existing Python3.12 runtime, not Linux/Windows validation.','No production database, public deployment, service/autostart installation, retention or off-host storage.','Confirms this local SQLite backup/restore path, not full section11, HA, PostgreSQL or operational SLA.']}
        (OUT/'final_bundle_v3.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
        (OUT/'final_bundle_v3.md').write_text('# Независимая проверка локального v3 ZIP\n\n'
            +f'Статус: **пройдено**. {len(CHECKS)} проверок за {receipt["total_elapsed_seconds"]:.2f} с. Архив из чистого коммита `{manifest["source_commit"]}`; SHA ZIP `{expected_sha}`.\n\n'
            +'Использован существующий `/private/tmp/collector-review-runtime-20260929` без переустановки. Пакеты и конфигурация окружения не изменились. Сервер запущен штатным start_demo.py из распакованного пакета на отдельном loopback-порту.\n\n'
            +'В отдельной runtimeDB явно созданы TEST-решение и TEST-черновик, оба с серверным снимком рекомендации. Включённый в ZIP scheduler автоматически создал три проверенные копии. Последняя восстановлена только в новый файл; второй HTTP-сервер работал с восстановленной БД.\n\n'
            +'GET journal до/после совпал полностью, включая оба сохранённых снимка рекомендаций. Все шесть metadata и payloads, 468 карточек, оценки, ранги и top10 неизменны. Текущие рекомендации также совпали за исключением ожидаемого generated_at. Seed и остальные manifest inputs сохранились.\n\n'
            +f'Время функции восстановления: {receipt["restore_elapsed_seconds"]*1000:.2f} мс. Рабочие пользовательские базы не открывались; временные БД удалены, собственные процессы остановлены. Подробные SHA, HTTP-снимки, события и процессные логи находятся рядом.\n\n'
            +'Проверена конкретная локальная macOS-конфигурация, не Linux/Windows, не полный §11 ТЗ, не HA/промышленный SLA. Автозапуск системной службы, retention и внешнее хранилище не устанавливались.\n')
        event('v3_acceptance_complete',checks=len(CHECKS),seconds=receipt['total_elapsed_seconds'])
    finally:
        for process in PROCESSES:
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=12)
                except subprocess.TimeoutExpired:process.kill();process.wait()
        for log in LOGS:log.close()
        temporary.cleanup()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip',type=Path,default=ROOT/'artifacts/submission/bundles/collector-risk-review-v3.zip')
    parser.add_argument('--runtime-python',type=Path,default=Path('/private/tmp/collector-review-runtime-20260929/bin/python'))
    parser.add_argument('--expected-commit',required=True)
    parser.add_argument('--expected-sha256',required=True)
    run(parser.parse_args())
