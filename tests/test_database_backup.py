"""Independent restore drills using only synthetic temporary SQLite databases."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from scripts import database_backup as backup


def create_fixture(path):
    """Keep the returned writer open so committed content remains in a live WAL."""
    writer = sqlite3.connect(path)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute("PRAGMA user_version=7")
    writer.execute("PRAGMA application_id=1280590658")
    writer.executescript('''
        CREATE TABLE tickets (id TEXT PRIMARY KEY, risk_id TEXT UNIQUE, note TEXT,
            created_at TEXT, status TEXT, entity_mode TEXT, risk_snapshot TEXT);
        CREATE TABLE feedback (id TEXT PRIMARY KEY, risk_id TEXT, entity_mode TEXT,
            decision TEXT, reason TEXT, operator TEXT, note TEXT, created_at TEXT, risk_snapshot TEXT);
        CREATE TABLE snapshots (id TEXT PRIMARY KEY, payload TEXT NOT NULL) WITHOUT ROWID;
        CREATE TABLE forecasts (id INTEGER PRIMARY KEY AUTOINCREMENT, object_id TEXT,
            score REAL, forecast_payload TEXT, model_bytes BLOB, unknown_value);
        CREATE INDEX forecast_object_idx ON forecasts(object_id);
        CREATE VIEW forecast_summary AS SELECT object_id,score FROM forecasts;
        CREATE TABLE forecast_audit (forecast_id INTEGER, action TEXT);
        CREATE TRIGGER future_forecast_insert AFTER INSERT ON forecasts
            BEGIN INSERT INTO forecast_audit VALUES (new.id,'created'); END;
        CREATE TABLE "future table with ""quotes""" ("value" TEXT);
    ''')
    payload = json.dumps({"forecast": {"issue_time": "2026-06-29T00:00:00+03:00", "model": "frozen",
                                      "minimum_lead_hours": 24},
                          "card": {"id": "synthetic-object", "score": .375,
                                   "explanation": ["Проверить датчик", "строка\nс 'кавычками'"],
                                   "observed": False, "missing": None}}, ensure_ascii=False)
    writer.execute("INSERT INTO tickets VALUES (?,?,?,?,?,?,?)",
                   ("ticket-1", "synthetic-object", "Осмотр, без отправки", "2026-09-29T00:00:00Z", "draft", "object", payload))
    writer.execute("INSERT INTO feedback VALUES (?,?,?,?,?,?,?,?,?)",
                   ("feedback-1", "synthetic-object", "object", "inspect", "needs_inspection", "Тестовый диспетчер",
                    "Проверочный текст", "2026-09-29T00:01:00Z", payload))
    writer.execute("INSERT INTO snapshots VALUES (?,?)", ("snapshot-1", payload))
    writer.execute("INSERT INTO forecasts(object_id,score,forecast_payload,model_bytes,unknown_value) VALUES (?,?,?,?,?)",
                   ("synthetic-object", .375, payload, b"\x00\xff\x01binary", None))
    writer.execute('INSERT INTO "future table with ""quotes""" VALUES (?)', ("неизвестная будущая таблица",))
    writer.commit()
    return writer, payload


class DatabaseBackupTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix="lct-backup-test-")
        self.root = Path(self.folder.name)
        self.source = self.root / "live.sqlite3"
        self.writer, self.payload = create_fixture(self.source)
        self.bundle = self.root / "backup"
        self.restored = self.root / "restored.sqlite3"

    def tearDown(self):
        self.writer.close()
        self.folder.cleanup()

    def assert_no_staging(self):
        self.assertEqual(list(self.root.glob(".lct-backup-*")), [])
        self.assertEqual(list(self.root.glob(".lct-restore-*")), [])

    def make_backup(self):
        return backup.backup_database(self.source, self.bundle)

    def test_live_wal_roundtrip_all_payloads_schema_and_future_tables(self):
        self.assertGreater(Path(str(self.source) + "-wal").stat().st_size, 0)
        # Uncommitted changes must never enter the snapshot.
        self.writer.execute("INSERT INTO snapshots VALUES (?,?)", ("uncommitted", "not saved"))
        copied = self.make_backup()
        self.writer.rollback()
        verified = backup.verify_backup(self.bundle)
        restored = backup.restore_database(self.bundle, self.restored)
        self.assertEqual(copied["source_journal_mode"], "wal")
        self.assertEqual(copied["inventory"], verified["inventory"])
        self.assertEqual(verified["inventory"], restored["inventory"])
        self.assertIn("forecasts", restored["inventory"]["user_tables"])
        self.assertEqual(restored["inventory"]["tables"]["snapshots"]["row_count"], 1)
        self.assertTrue(restored["inventory"]["tables"]["sqlite_sequence"]["internal"])
        with closing(sqlite3.connect(self.restored)) as conn:
            self.assertEqual(conn.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            self.assertEqual(conn.execute("SELECT risk_snapshot FROM tickets").fetchone()[0], self.payload)
            self.assertEqual(conn.execute("SELECT decision,reason,risk_snapshot FROM feedback").fetchone(),
                             ("inspect", "needs_inspection", self.payload))
            self.assertEqual(conn.execute("SELECT payload FROM snapshots").fetchone()[0], self.payload)
            self.assertEqual(conn.execute("SELECT score,forecast_payload,model_bytes,unknown_value FROM forecasts").fetchone(),
                             (.375, self.payload, b"\x00\xff\x01binary", None))
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 7)
            self.assertEqual(conn.execute("PRAGMA application_id").fetchone()[0], 1280590658)
            self.assertEqual(conn.execute("PRAGMA journal_mode").fetchone()[0], "delete")
            # Preserved trigger and autoincrement state work after restoration.
            conn.execute("INSERT INTO forecasts(object_id) VALUES ('new-synthetic-object')")
            self.assertEqual(conn.execute("SELECT max(id) FROM forecasts").fetchone()[0], 2)
            self.assertEqual(conn.execute("SELECT count(*) FROM forecast_audit").fetchone()[0], 2)
        self.assert_no_staging()

    def test_wal_writer_can_commit_after_snapshot_is_pinned(self):
        original_copy = backup.copy_online

        def concurrent_commit(source, destination, deadline):
            with closing(sqlite3.connect(self.source)) as other_writer:
                other_writer.execute("INSERT INTO snapshots VALUES ('late-commit','after the snapshot')")
                other_writer.commit()
            return original_copy(source, destination, deadline)

        with patch.object(backup, "copy_online", side_effect=concurrent_commit):
            copied = self.make_backup()
        self.assertEqual(self.writer.execute("SELECT count(*) FROM snapshots").fetchone()[0], 2)
        self.assertEqual(copied["inventory"]["tables"]["snapshots"]["row_count"], 1)
        backup.restore_database(self.bundle, self.restored)
        with closing(sqlite3.connect(self.restored)) as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM snapshots").fetchone()[0], 1)
        self.assert_no_staging()

    def test_existing_working_database_bundle_and_dangling_symlink_are_refused(self):
        self.make_backup()
        original = backup.sha256(self.source)
        wal_original = backup.sha256(Path(str(self.source) + "-wal"))
        with self.assertRaises(FileExistsError):
            backup.restore_database(self.bundle, self.source)
        self.assertEqual(backup.sha256(self.source), original)
        self.assertEqual(backup.sha256(Path(str(self.source) + "-wal")), wal_original)
        with self.assertRaises(FileExistsError):
            self.make_backup()
        self.restored.symlink_to(self.root / "missing-target")
        with self.assertRaises(FileExistsError):
            backup.restore_database(self.bundle, self.restored)
        self.assertTrue(self.restored.is_symlink())
        self.assert_no_staging()

    def test_orphan_destination_wal_is_refused(self):
        self.make_backup()
        sidecar = Path(str(self.restored) + "-wal")
        sidecar.write_bytes(b"do not overwrite")
        with self.assertRaises(FileExistsError):
            backup.restore_database(self.bundle, self.restored)
        self.assertEqual(sidecar.read_bytes(), b"do not overwrite")
        self.assertFalse(self.restored.exists())

    def test_corrupt_and_non_sqlite_backups_are_rejected(self):
        self.make_backup()
        database = self.bundle / backup.DB_NAME
        database.write_bytes(b"not sqlite; intentionally damaged")
        with self.assertRaises(backup.BackupError):
            backup.restore_database(self.bundle, self.restored)
        # Even an updated checksum cannot bypass the SQLite/schema/payload validation.
        manifest_path = self.bundle / backup.MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text())
        manifest.update(database_bytes=database.stat().st_size, database_sha256=backup.sha256(database))
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(sqlite3.DatabaseError):
            backup.restore_database(self.bundle, self.restored)
        with self.assertRaises(sqlite3.DatabaseError):
            backup.backup_database(database, self.root / "must-not-exist")
        self.assertFalse(self.restored.exists())
        self.assertFalse((self.root / "must-not-exist").exists())
        self.assert_no_staging()

    def test_payload_change_with_recomputed_file_checksum_is_rejected(self):
        self.make_backup()
        database = self.bundle / backup.DB_NAME
        with closing(sqlite3.connect(database)) as conn:
            conn.execute("UPDATE feedback SET decision='alarm_confirmed'")
            conn.commit()
        manifest_path = self.bundle / backup.MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text())
        manifest.update(database_bytes=database.stat().st_size, database_sha256=backup.sha256(database))
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(backup.BackupError):
            backup.restore_database(self.bundle, self.restored)
        self.assertFalse(self.restored.exists())
        self.assert_no_staging()

    def test_structural_page_damage_with_valid_sqlite_header_is_rejected(self):
        self.make_backup()
        database = self.bundle / backup.DB_NAME
        raw = bytearray(database.read_bytes())
        self.assertEqual(raw[:16], b"SQLite format 3\x00")
        page_size = int.from_bytes(raw[16:18], "big") or 65536
        self.assertGreater(len(raw), page_size + 24)
        raw[page_size:page_size + 24] = b"\xff" * 24
        database.write_bytes(raw)
        manifest_path = self.bundle / backup.MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text())
        manifest["database_sha256"] = backup.sha256(database)
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises((sqlite3.DatabaseError, backup.BackupError)):
            backup.restore_database(self.bundle, self.restored)
        self.assertFalse(self.restored.exists())
        self.assert_no_staging()

    def test_restore_failure_removes_only_its_own_temporary_files(self):
        self.make_backup()
        unrelated = self.root / ".lct-restore-unrelated.sqlite3"
        unrelated.write_bytes(b"must remain")
        with patch.object(backup, "copy_online", side_effect=backup.BackupError("injected copy failure")):
            with self.assertRaises(backup.BackupError):
                backup.restore_database(self.bundle, self.restored)
        self.assertFalse(self.restored.exists())
        self.assertEqual(unrelated.read_bytes(), b"must remain")
        self.assertEqual(list(self.root.glob(".lct-restore-*")), [unrelated])

    def test_atomic_publish_does_not_overwrite_a_concurrent_destination(self):
        self.make_backup()
        original_rename = backup.atomic_rename_new

        def create_racing_file(source, destination):
            Path(destination).write_bytes(b"created by another operation")
            return original_rename(source, destination)

        with patch.object(backup, "atomic_rename_new", side_effect=create_racing_file):
            with self.assertRaises(FileExistsError):
                backup.restore_database(self.bundle, self.restored)
        self.assertEqual(self.restored.read_bytes(), b"created by another operation")
        self.assert_no_staging()


if __name__ == "__main__":
    unittest.main()
