"""Summarize both fixed-trace repetitions without picking a favorable run."""
from pathlib import Path
import json
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'reports/runtime_comparison'


def main():
    results={key:json.loads((OUTPUT/key/'summary.json').read_text()) for key in ['baseline','optimized']}
    assert all(x['passed'] and x['source_files_unchanged'] for x in results.values())
    rows=[]
    for scenario in ['read_only','mixed']:
        for users in [1,20]:
            entry={'scenario':scenario,'users':users}
            for key, result in results.items():
                selected=[r for r in result['series'] if r['scenario']==scenario and r['users']==users]
                assert len(selected)==2 and all(r['requests']==1200 and r['errors']==0 for r in selected)
                entry[key]={'requests_per_second':statistics.mean(r['requests_per_second'] for r in selected),
                            **{f'p{p}_ms':statistics.mean(r['latency'][f'p{p}_ms'] for r in selected) for p in [50,95,99]}}
            entry['throughput_change_percent']=(entry['optimized']['requests_per_second']/entry['baseline']['requests_per_second']-1)*100
            entry['p95_change_percent']=(entry['optimized']['p95_ms']/entry['baseline']['p95_ms']-1)*100
            rows.append(entry)
    report={'aggregation':'Arithmetic mean of each repetition metric; not pooled quantiles or confidence intervals',
            'rows':rows,'requests_per_version':9600,'requests_total':19200,'errors_total':0,
            'recommendation_scope':'status checked, plus POST card/run identity; full recommendation equality not claimed',
            'official_no_degradation_requirement_proven':False}
    (OUTPUT/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines=['# Работа архива и журнала: одинаковая нагрузка до и после исправления','',
           'Проверены две версии API, каждый раз на новой копии одинакового seed с шестью выпусками. Всего **19 200 запросов, 0 ошибок**. В каждой смешанной серии сохранены ровно 120 решений и 120 новых черновиков; их ID совпали с подтверждёнными ответами сервера. Все карточки снимков и шесть исходных payload совпали точно. Статусы рекомендаций проверены; полного сравнения всех их полей этот нагрузочный тест не выполняет.','',
           '## Изменение кода','',
           '- Чтение готового журнала больше не выполняет `BEGIN IMMEDIATE`. Проверка схемы остаётся при каждом соединении; миграция выполняется под блокировкой с повторной проверкой столбцов. Кэша готовности схемы в процессе нет.',
           '- Payload и метаданные выпуска читаются из одной проверенной строки. Проверка контрольной суммы, схемы и привязки карточек сохранена; дублирующее чтение убрано.',
           '- Режим SQLite DELETE, количество серверных процессов, модели и состав запросов не менялись. Поддержка чтения при RESERVED lock подтверждена отдельным тестом, который на прежней версии падал. Чтение при EXCLUSIVE lock этим не обещается.','',
           '## Результаты','',
           'В ячейках «до → после». Время — миллисекунды. Каждая цифра является средним показателей двух отдельных повторов; усреднение p95 не является p95 объединённой выборки.','',
           '| Сценарий | Клиенты | Запросов/с | p50, мс | p95, мс | p99, мс |',
           '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        values=[f"{r['baseline'][k]:.1f} → {r['optimized'][k]:.1f}" for k in ['requests_per_second','p50_ms','p95_ms','p99_ms']]
        lines.append('| '+('Чтение' if r['scenario']=='read_only' else 'Чтение и запись')+f" | {r['users']} | "+' | '.join(values)+' |')
    lines+=['','При 20 клиентах хвост задержек сократился, но медиана выросла. В этих сериях уменьшение длительных ожиданий не означает ускорение каждого запроса. Наблюдаемая пропускная способность выросла примерно на 12–13% при 20 клиентах; два коротких повтора на общей машине не устанавливают стабильный промышленный эффект.','',
            '### После исправления: 20 клиентов относительно одного','',
            'Требование ТЗ «без деградации» **не подтверждено**: при насыщенной нагрузке без пауз p95 выше, чем у одного клиента. Проверены 20 конкурентных запросов, а не 20 последовательных пользовательских сценариев. Численного допустимого увеличения задержки ТЗ не задаёт; искусственный SLA не вводится.','',
            '| Сценарий / маршрут | p95 при 1, мс | p95 при 20, мс | Отношение |',
            '|---|---:|---:|---:|']
    for scenario in ['read_only','mixed']:
        samples=[x for x in results['optimized']['series'] if x['scenario']==scenario]
        for route in sorted(samples[0]['by_route']):
            a=statistics.mean(x['by_route'][route]['p95_ms'] for x in samples if x['users']==1)
            b=statistics.mean(x['by_route'][route]['p95_ms'] for x in samples if x['users']==20)
            lines.append(f'| {scenario} / {route} | {a:.1f} | {b:.1f} | {b/a:.2f}× |')
    lines+=['','## Метод и ограничения','',
            '120 циклов × 10 операций на серию; два повтора, порядок 1→20 и 20→1. Read-only содержит только GET; mixed — 8 GET и два POST. Используются явные run_id всех шести выпусков. Порядок завершения и промежуточный размер журнала при конкуренции неизбежно отличаются; итоговые множества запросов и новых записей совпадают. POST не повторяются автоматически. Старые архивы и рабочая БД пользователя не используются.','',
            'Один процесс Uvicorn, постоянное HTTP-соединение на клиента, без пауз между запросами. Latency включает чтение тела и JSON decode, но не последующую проверку значений; throughput включает работу проверяющего клиента. Клиент и сервер делят CPU. Max inflight включает короткую клиентскую валидацию. Длительности серий 4–12 секунд; p99 по маршруту на 120–240 наблюдениях нестабилен. Временные БД удалены после прямой проверки; полные записи для последующего аудита не сохранены. CSV содержит подтверждённые ID и сырые задержки.','',
            'ML inference, реальная сеть, TLS/LDAP/RBAC, браузерная отрисовка, Linux, длительная работа и нагрузка растущего многолетнего журнала здесь не проверялись. Текущее окно feedback/journal ограничено 200 записями, tickets возвращает все строки; масштабирование истории остаётся отдельной задачей.','',
            'В исправлении не менялся WAL: одновременность читателей и писателя и ограничение одного писателя описаны в [документации SQLite](https://www.sqlite.org/wal.html). Изменение режима журналирования требует отдельного измерения и здесь не использовалось для улучшения цифр.','',
            '## Доказательства и воспроизведение','',
            '[Предварительный review](protocol_review.md), [review хранения](storage_review.md), [исправление названия доказательства](claim_correction.json), [baseline](baseline/summary.json), [после](optimized/summary.json), [170 тестов](full_tests.log). В каждой папке версии сохранены протокол/SHA, оба trace, восемь CSV и серверные логи. Единственная правка harness между сериями исправила слишком широкую подпись результата; workload, валидаторы и таймеры не менялись. Исходники API между версиями изменены намеренно.','',
            '```sh',
            '.venv/bin/python scripts/compare_runtime.py --cycles 120 --repeats 2 --output reports/runtime_comparison/new-run',
            '.venv/bin/python scripts/summarize_runtime_comparison.py',
            '```','',
            'Для исторического baseline нужен API из коммита `5f16c96`; текущий API — исправленная версия. Harness требует локальный v2 ZIP (путь можно задать `--bundle`); новый output не должен существовать. Сводный скрипт читает сохранённые папки baseline/optimized. Публичный Git не содержит предоставленных данных/весов/ZIP.']
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
