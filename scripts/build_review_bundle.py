"""Build an explicitly local review kit. Never upload supplied data or weights.

Includes only named model/data files, code, selected manifests, and immutable
forecast releases. Dispatcher feedback and tickets are deliberately not copied.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.forecast_store import list_runs, load_run, publish_run

ASSETS = [
    'data/interim/daily_2025.parquet', 'data/interim/daily_2026.parquet',
    'data/raw/справочник_каналов_датчиков.csv', 'data/raw/справочник_объектов_диспетчер.csv',
    'models/object_onset_experiment/refit/hurdle_15.joblib', 'models/object_risk_probe.txt',
    'data/app/object_risk_demo.json', 'data/app/risk_demo.json',
    'reports/object_risk_probe/configuration.json',
    'reports/object_onset_experiment/refit.json', 'reports/object_onset_experiment/selection.json',
    'reports/object_onset_experiment/protocol.md',
    'requirements-runtime.txt', 'requirements-inference.txt',
    'docs/onset_demo.md', 'docs/object_demo.md', 'docs/local_demo.md',
]

VERIFY = '''"""Verify immutable bundle inputs; new runtime files are not inputs."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def verify():
    manifest=json.loads((ROOT/'bundle_manifest.json').read_text())
    for name,expected in manifest['files'].items():
        path=ROOT/name
        if not path.is_file() or path.is_symlink() or ROOT not in path.resolve().parents:
            raise ValueError('Missing or unsafe bundle input: '+name)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected['sha256']:
            raise ValueError('Changed bundle input: '+name)
    return manifest
if __name__=='__main__':
    result=verify()
    print(json.dumps({'verified_files':len(result['files']),'source_commit':result['source_commit']},indent=2))
'''

LAUNCH = '''"""Start the local retrospective demo; no public bind or external writes."""
import argparse,os,shutil,tempfile
from pathlib import Path
from verify_bundle import verify
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    if not 1<=args.port<=65535:parser.error('Port must be between 1 and 65535')
    verify()
    runtime=ROOT/'data/app/dispatch.sqlite3'
    if not runtime.exists():
        with tempfile.TemporaryDirectory(prefix='.seed-',dir=runtime.parent) as stage:
            prepared=Path(stage)/'dispatch.sqlite3'
            shutil.copyfile(ROOT/'data/app/forecast_seed.sqlite3',prepared)
            try:os.link(prepared,runtime)
            except FileExistsError:pass  # Another startup atomically published the complete seed.
    os.environ['LCT_TICKET_DB']=str(runtime)
    os.chdir(ROOT)
    import uvicorn
    uvicorn.run('api.main:app',host='127.0.0.1',port=args.port)
'''

INSTRUCTIONS = '''# Collector Risk - локальный комплект проверки

Архивная демонстрация обеих объектных целей и прежнего канального режима.
Команда и конкурсная сдача этим комплектом не подтверждаются.

## Быстрый запуск (Python 3.12)

Из распакованной папки collector-risk-review:

    python3.12 -m venv .venv
    .venv/bin/python -m pip install -r requirements-runtime.txt
    .venv/bin/python verify_bundle.py
    .venv/bin/python start_demo.py --port 8765

Откройте http://127.0.0.1:8765/ . Для новой цели выберите «Начало серии».
Windows использует .venv\\Scripts\\python.exe вместо .venv/bin/python.
Загрузка зависимостей требует доступа к PyPI. Сам сервер читает локальные файлы.

## Пересчёт сохранённых моделей

    .venv/bin/python -m pip install -r requirements-inference.txt
    .venv/bin/python src/serving/object_onset_forecast.py --database data/app/recomputed_onset.sqlite3
    .venv/bin/python src/serving/object_forecast.py --database data/app/recomputed_any.sqlite3

Обе команды используют дневные агрегаты и фиксированные веса без обучения.
Проверка исходников и контрольных сумм выполняется до загрузки onset-модели.
Для пересчёта моделей на macOS нужен установленный OpenMP runtime для LightGBM
(в проверенной среде Homebrew libomp); Linux/Windows отдельно не проверены.
Код исследовательских экспериментов включён, но этот комплект не содержит все
артефакты обучения и восемь исходных архивов. Полное переобучение не заявлено.

## Состав и ограничения

Два дневных набора 2025/2026, два справочника, две замороженные модели,
шесть неизменяемых выпусков и исходные демонстрационные карточки. Журналы
решений и заявки из рабочего каталога не копируются. При первом запуске
forecast_seed.sqlite3 копируется в отдельную dispatch.sqlite3; последующие
запуски сохраняют локальные тестовые решения. Используйте явные тестовые пометки.

Начало серии: нет зарегистрированной тревоги в D+1, есть в D+2; доступны
данные только до конца D. Ровно 10 объектов в каждом выпуске. Нет записей
не означает исправность. Score не является вероятностью физической поломки.
Модель исследовательская; прогнозы исторические. TLS, корпоративный вход,
рабочий поток и внешняя отправка заявок отсутствуют.

Исходный код: https://github.com/AnnaRazina10/lct-collector-risk

Этот комплект содержит предоставленные данные и веса, поэтому остаётся
локальным и исключён из Git. Его публичное размещение не выполнялось.
bundle_manifest.json фиксирует состав и SHA-256, но не является цифровой подписью.
'''


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_copy(source_root, relative, destination):
    source = source_root/relative
    if source.is_symlink() or source_root.resolve() not in source.resolve().parents or not source.is_file():
        raise ValueError('Missing or unsafe input: '+relative)
    target = destination/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def build(output, forecast_database):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a new output; existing bundle is not overwritten')
    source_commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    tracked = subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    selected = sorted(set(ASSETS + [p for p in tracked if p.startswith(('api/','web/','src/'))]))
    with tempfile.TemporaryDirectory(prefix='collector-review-') as tmp:
        folder = Path(tmp)/'collector-risk-review'; folder.mkdir()
        for name in selected:
            safe_copy(ROOT, name, folder)
        (folder/'verify_bundle.py').write_text(VERIFY)
        (folder/'start_demo.py').write_text(LAUNCH)
        (folder/'START_HERE.md').write_text(INSTRUCTIONS)
        seed = folder/'data/app/forecast_seed.sqlite3'
        releases = list_runs(forecast_database, entity_mode='object', limit=500)
        if not releases:
            raise ValueError('No immutable forecast releases to include')
        for release in releases:
            publish_run(seed, load_run(forecast_database, release['run_id'], entity_mode='object'))
        manifest = {'created_at':datetime.now(timezone.utc).isoformat(), 'source_commit':source_commit,
            'working_tree_changes':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)),
            'purpose':'local review and frozen inference; not complete retraining or public deployment',
            'dispatcher_records_included':False, 'release_ids':[r['run_id'] for r in releases],
            'files':{str(p.relative_to(folder)):{'sha256':digest(p),'bytes':p.stat().st_size}
                     for p in sorted(folder.rglob('*')) if p.is_file()}}
        (folder/'bundle_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        output.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(output,'x',ZIP_DEFLATED,compresslevel=6) as archive:
            for path in sorted(folder.rglob('*')):
                if path.is_file(): archive.write(path,str(path.relative_to(Path(tmp))))
    return {'path':str(output),'sha256':digest(output),'bytes':output.stat().st_size,
            'source_commit':source_commit,'files':len(manifest['files'])+1,'releases':len(releases),
            'dispatcher_records_included':False}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/submission/bundles/collector-risk-review.zip')
    parser.add_argument('--forecast-database',type=Path,default=ROOT/'data/app/onset_demo.sqlite3')
    args=parser.parse_args()
    print(json.dumps(build(args.output,args.forecast_database),ensure_ascii=False,indent=2))
