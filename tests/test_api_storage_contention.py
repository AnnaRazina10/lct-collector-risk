"""Journal reads must not compete for the writer reservation after setup."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from api import main


class JournalStorageContentionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.db=Path(self.tmp.name)/'journal.sqlite3'
        self.binding=patch.object(main,'DB',self.db)
        self.binding.start()

    def tearDown(self):
        self.binding.stop();self.tmp.cleanup()

    def test_ready_read_works_while_another_writer_holds_reserved_lock(self):
        with main.connection() as conn:
            conn.execute("INSERT INTO tickets VALUES ('t','r','committed','2026','draft','object','{}')")
        writer=sqlite3.connect(self.db)
        real_connect=sqlite3.connect
        try:
            writer.execute('BEGIN IMMEDIATE')
            writer.execute("UPDATE tickets SET note='not committed'")
            with patch.object(main.sqlite3,'connect',side_effect=lambda *a,**k:real_connect(*a,**{**k,'timeout':.05})):
                result=main.tickets(mode='object')
            self.assertEqual(result['tickets'][0]['note'],'committed')
        finally:
            writer.rollback();writer.close()

    def test_concurrent_first_use_retains_every_committed_write(self):
        def save(index):
            with main.connection() as conn:
                conn.execute('INSERT INTO feedback VALUES (?,?,?,?,?,?,?,?,?)',
                    (str(index),'risk','object','monitor','other','TEST','note','2026','{}'))
        with ThreadPoolExecutor(max_workers=12) as pool:list(pool.map(save,range(24)))
        with sqlite3.connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM feedback').fetchone()[0],24)
            self.assertEqual(conn.execute('PRAGMA integrity_check').fetchone()[0],'ok')

    def test_legacy_migration_preserves_channel_draft(self):
        with sqlite3.connect(self.db) as conn:
            conn.execute('CREATE TABLE tickets (id TEXT PRIMARY KEY, risk_id TEXT UNIQUE, note TEXT, created_at TEXT, status TEXT)')
            conn.execute("INSERT INTO tickets VALUES ('t','r','old note','2026','draft')")
        record=main.tickets()['tickets'][0]
        self.assertEqual(record['note'],'old note')
        self.assertEqual(record['entity_mode'],'channel')
        self.assertEqual(record['risk_snapshot'],{})

    def test_failed_write_rolls_back_and_leaves_schema_usable(self):
        with self.assertRaisesRegex(RuntimeError,'cancel test'):
            with main.connection() as conn:
                conn.execute("INSERT INTO tickets VALUES ('t','r','note','2026','draft','object','{}')")
                raise RuntimeError('cancel test')
        self.assertEqual(main.tickets()['tickets'],[])


if __name__=='__main__':unittest.main()
