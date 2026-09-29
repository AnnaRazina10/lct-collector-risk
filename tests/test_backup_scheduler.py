"""Scheduler acceptance with real subprocess locks and isolated SQLite sources."""
from contextlib import closing
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts import backup_scheduler as scheduler
from scripts import database_backup as backup

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/backup_scheduler.py"


@unittest.skipUnless(os.name == "posix", "Real flock tests require POSIX")
class BackupSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix="lct-scheduler-test-")
        self.root = Path(self.folder.name)
        self.source = self.root / "source.sqlite3"
        self.output = self.root / "backups"
        self.processes = []
        self.payload = json.dumps({"forecast": {"id": "test-run", "score": .25},
            "recommendation": {"ruleset_version": "1.0.0", "steps": ["Проверить поступление данных"]},
            "note": "Тестовая локальная запись, без внешней отправки"}, ensure_ascii=False)
        with closing(sqlite3.connect(self.source)) as conn:
            conn.executescript("CREATE TABLE forecasts(id TEXT PRIMARY KEY,payload TEXT);"
                               "CREATE TABLE decisions(id TEXT PRIMARY KEY,snapshot TEXT);")
            conn.execute("INSERT INTO forecasts VALUES (?,?)", ("run", self.payload))
            conn.execute("INSERT INTO decisions VALUES (?,?)", ("decision", self.payload))
            conn.commit()

    def tearDown(self):
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=6)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=3)
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()
        self.folder.cleanup()

    def args(self, interval=.1, max_runs=None):
        args = ["--source", str(self.source), "--output-dir", str(self.output),
                "--interval", str(interval), "--timeout", "2"]
        if max_runs is not None:
            args.extend(["--max-runs", str(max_runs)])
        return args

    def start(self, interval=.1, max_runs=None, *, delay_backup=None):
        command = [sys.executable, str(SCRIPT)]
        if delay_backup is not None:
            code = ("import time; from scripts import backup_scheduler as s; "
                    "original=s.database_backup.backup_database; "
                    f"s.database_backup.backup_database=lambda *a,**k: (time.sleep({delay_backup}),original(*a,**k))[1]; "
                    "raise SystemExit(s.main())")
            command = [sys.executable, "-c", code]
        process = subprocess.Popen(command+self.args(interval, max_runs), cwd=ROOT,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.processes.append(process)
        return process

    def read_status(self):
        return json.loads((self.output / scheduler.STATUS_NAME).read_text())

    def wait_status(self, predicate, process=None, timeout=6):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            try:
                status = self.read_status()
            except FileNotFoundError:
                status = None
            if status is not None and predicate(status):
                return status
            if process is not None and process.poll() is not None:
                self.fail(f"Child exited {process.returncode}: {process.stderr.read()} ; status={status}")
            time.sleep(.005)
        self.fail(f"Timed out waiting for status: {status}")

    def finish(self, process, expected=0):
        stdout, stderr = process.communicate(timeout=8)
        self.assertEqual(process.returncode, expected, stderr or stdout)
        return json.loads(stdout) if stdout.strip() else None

    def test_three_periodic_backups_restore_payload_and_never_overwrite(self):
        before = backup.sha256(self.source)
        status = self.finish(self.start(max_runs=3))
        self.assertEqual(status["counters"], {"attempts": 3, "successes": 3, "failures": 0, "interrupted": 0})
        self.assertEqual(status["last_attempt"]["status"], "succeeded")
        self.assertEqual(status["state"], "stopped")
        self.assertEqual(status["stop_reason"], "max_runs")
        self.assertEqual(backup.sha256(self.source), before)
        bundles = sorted(self.output.glob("backup-*"))
        self.assertEqual(len(bundles), 3)
        self.assertEqual(len({item.name for item in bundles}), 3)
        originals = {str(item): backup.sha256(item / backup.DB_NAME) for item in bundles}
        for item in bundles:
            self.assertTrue(backup.verify_backup(item)["passed"])
        restored = self.root / "restored.sqlite3"
        backup.restore_database(status["last_success"]["destination"], restored)
        with closing(sqlite3.connect(restored)) as conn:
            self.assertEqual(conn.execute("SELECT payload FROM forecasts").fetchone()[0], self.payload)
            self.assertEqual(conn.execute("SELECT snapshot FROM decisions").fetchone()[0], self.payload)
        again = self.finish(self.start(max_runs=1))
        self.assertEqual(again["session_attempts"], 1)
        self.assertEqual(again["counters"]["successes"], 4)
        self.assertEqual(len(list(self.output.glob("backup-*"))), 4)
        for path, checksum in originals.items():
            self.assertEqual(backup.sha256(Path(path) / backup.DB_NAME), checksum)

    def test_real_competing_process_refused_without_touching_status(self):
        owner = self.start(interval=60)
        held = self.wait_status(lambda s: s["state"] == "waiting", owner)
        original = (self.output / "status.json").read_bytes()
        rival = self.start(max_runs=1)
        stdout, stderr = rival.communicate(timeout=5)
        self.assertEqual(rival.returncode, 3, stdout+stderr)
        self.assertEqual(json.loads(stderr)["status"], "locked")
        self.assertEqual((self.output / "status.json").read_bytes(), original)
        self.assertEqual(len(list(self.output.glob("backup-*"))), 1)
        owner.terminate()
        ended = self.finish(owner, 143)
        self.assertEqual(ended["stop_reason"], "signal:SIGTERM")
        self.assertEqual(ended["last_success"], held["last_success"])
        self.assertTrue((self.output / scheduler.LOCK_NAME).is_file())
        self.assertEqual(self.finish(self.start(max_runs=1))["counters"]["successes"], 2)

    def test_source_failure_preserves_last_success_then_recovers(self):
        process = self.start(interval=.2)
        first = self.wait_status(lambda s: s["state"] == "waiting" and s["session_successes"] == 1, process)
        missing = self.root / "source_temporarily_hidden.sqlite3"
        self.source.rename(missing)
        failed = self.wait_status(lambda s: s["state"] == "waiting" and s["session_failures"] >= 1, process)
        self.assertEqual(failed["last_attempt"]["status"], "failed")
        self.assertEqual(failed["last_success"], first["last_success"])
        self.assertEqual(failed["last_error"]["type"], "FileNotFoundError")
        self.assertTrue(Path(first["last_success"]["destination"]).is_dir())
        missing.rename(self.source)
        recovered = self.wait_status(lambda s: s["state"] == "waiting" and s["session_successes"] >= 2, process)
        self.assertEqual(recovered["last_attempt"]["status"], "succeeded")
        self.assertEqual(recovered["last_error"], failed["last_error"])
        self.assertNotEqual(recovered["last_success"]["destination"], first["last_success"]["destination"])
        process.terminate(); self.finish(process, 143)
        restart = self.finish(self.start(max_runs=1))
        self.assertEqual(restart["session_failures"], 0)
        self.assertGreaterEqual(restart["counters"]["failures"], 1)
        self.assertEqual(restart["last_error"], failed["last_error"])

    def test_bounded_failed_session_returns_nonzero_and_keeps_old_success(self):
        first = self.finish(self.start(max_runs=1))
        self.source.write_bytes(b"not a sqlite database")  # Only this synthetic fixture.
        failed = self.finish(self.start(max_runs=2), 1)
        self.assertEqual(failed["session_failures"], 2)
        self.assertEqual(failed["last_success"], first["last_success"])
        self.assertEqual(len(list(self.output.glob("backup-*"))), 1)
        self.assertEqual(list(self.output.glob(".lct-backup-*")), [])

    def test_sigint_interrupts_wait_and_no_extra_backup_starts(self):
        process = self.start(interval=60)
        first = self.wait_status(lambda s: s["state"] == "waiting", process)
        process.send_signal(signal.SIGINT)
        stopped = self.finish(process, 130)
        self.assertEqual(stopped["stop_reason"], "signal:SIGINT")
        self.assertEqual(stopped["counters"], first["counters"])
        self.assertIsNone(stopped["next_attempt_at_utc"])

    def test_sigterm_during_active_backup_finishes_it_then_stops(self):
        process = self.start(interval=.05, delay_backup=.25)
        self.wait_status(lambda s: s["state"] == "backing_up", process)
        process.terminate()
        stopped = self.finish(process, 143)
        self.assertEqual(stopped["session_attempts"], 1)
        self.assertEqual(stopped["last_attempt"]["status"], "succeeded")
        self.assertTrue(backup.verify_backup(stopped["last_success"]["destination"])["passed"])

    def test_os_releases_lock_after_sigkill_and_restart_records_interrupted_attempt(self):
        process = self.start(interval=60, delay_backup=2)
        self.wait_status(lambda s: s["state"] == "backing_up", process)
        process.kill(); process.wait(timeout=3)
        self.assertEqual(process.returncode, -signal.SIGKILL)
        restored = self.finish(self.start(max_runs=1))
        self.assertEqual(restored["counters"], {"attempts": 2, "successes": 1, "failures": 0, "interrupted": 1})
        self.assertEqual(restored["last_error"]["type"], "InterruptedAttempt")

    def test_slow_attempts_skip_slots_without_catchup_burst(self):
        process = self.start(interval=.05, max_runs=3, delay_backup=.13)
        status = self.finish(process)
        self.assertGreaterEqual(status["skipped_slots"], 6)
        self.assertEqual(status["session_attempts"], 3)
        self.assertEqual(len(list(self.output.glob("backup-*"))), 3)

    def test_real_live_wal_backup_includes_committed_but_not_uncommitted_data(self):
        with closing(sqlite3.connect(self.source)) as writer:
            writer.execute("PRAGMA journal_mode=WAL")
            writer.execute("PRAGMA wal_autocheckpoint=0")
            writer.execute("INSERT INTO forecasts VALUES ('wal-commit', ?)", (self.payload,)); writer.commit()
            writer.execute("INSERT INTO forecasts VALUES ('uncommitted', 'must not appear')")
            status = self.finish(self.start(max_runs=1))
            self.assertEqual(status["last_success"]["source_journal_mode"], "wal")
            copied = backup.verify_backup(status["last_success"]["destination"])
            self.assertEqual(copied["inventory"]["tables"]["forecasts"]["row_count"], 2)
            writer.rollback()

    def test_invalid_config_rejected_before_creating_output(self):
        for field, values in {"interval": [0, -1, float("nan"), float("inf"), 1e-20, 10**1000],
                              "timeout": [0, -1, float("nan"), float("inf")],
                              "max_runs": [0, -1, 1.5, True]}.items():
            for value in values:
                kwargs = {"interval": .1, "timeout": 2, "max_runs": 1}; kwargs[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    scheduler.run_scheduler(self.source, self.output, **kwargs)
                self.assertFalse(self.output.exists())
        result = subprocess.run([sys.executable, str(SCRIPT), *self.args(), "--interval", "nan"],
                                cwd=ROOT, capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.output.exists())

    def test_source_inside_output_refused_before_writes(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            scheduler.run_scheduler(self.source, self.root, max_runs=1)
        self.assertFalse((self.root / scheduler.STATUS_NAME).exists())

    def test_corrupt_status_or_different_source_refused_without_overwrite(self):
        first = self.finish(self.start(max_runs=1))
        status_path = self.output / scheduler.STATUS_NAME
        original = status_path.read_bytes()
        other = self.root / "other.sqlite3"
        other.write_bytes(self.source.read_bytes())
        with self.assertRaises(scheduler.SchedulerError):
            scheduler.run_scheduler(other, self.output, max_runs=1)
        self.assertEqual(status_path.read_bytes(), original)
        status_path.write_text("{broken-json")
        with self.assertRaises(ValueError):
            scheduler.run_scheduler(self.source, self.output, max_runs=1)
        self.assertEqual(status_path.read_text(), "{broken-json")
        self.assertTrue(backup.verify_backup(first["last_success"]["destination"])["passed"])

    def test_atomic_status_failure_preserves_previous_json_and_cleans_own_temp(self):
        self.output.mkdir()
        first = {"state": "old", "last_success": {"destination": "preserve"}}
        scheduler.write_status(self.output, first)
        previous = (self.output / scheduler.STATUS_NAME).read_bytes()
        with patch.object(scheduler.os, "replace", side_effect=OSError("injected status error")):
            with self.assertRaises(OSError):
                scheduler.write_status(self.output, {"state": "new"})
        self.assertEqual((self.output / scheduler.STATUS_NAME).read_bytes(), previous)
        self.assertEqual(list(self.output.glob(".status-*.tmp")), [])

    def test_symlink_status_is_not_followed(self):
        self.output.mkdir()
        unrelated = self.root / "unrelated.json"; unrelated.write_text('{"preserve":true}')
        (self.output / scheduler.STATUS_NAME).symlink_to(unrelated)
        with self.assertRaises(OSError):
            scheduler.run_scheduler(self.source, self.output, max_runs=1)
        self.assertEqual(unrelated.read_text(), '{"preserve":true}')

    def test_monotonic_deadline_skips_elapsed_slots_not_wall_clock(self):
        for previous, finished, interval, expected, skipped in [
            (0, 1, 10, 10, 0), (0, 10, 10, 20, 1), (0, 35, 10, 40, 3),
            (1000, 1025, 10, 1030, 2)]:
            with self.subTest(finished=finished):
                actual, count = scheduler.next_deadline(previous, finished, interval)
                self.assertEqual((actual, count), (expected, skipped))
                self.assertGreater(actual, finished)


if __name__ == "__main__":
    unittest.main()
