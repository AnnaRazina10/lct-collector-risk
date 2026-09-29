#!/usr/bin/env python3
"""Explicit local SQLite backup scheduler; no service installation or retention."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import signal
import stat
import sys
import tempfile
import threading
import time
from uuid import uuid4

try:
    import fcntl
except ImportError:  # Fail explicitly on unsupported platforms.
    fcntl = None

try:
    from . import database_backup
except ImportError:  # python scripts/backup_scheduler.py
    import database_backup

FORMAT_VERSION = 1
LOCK_NAME = ".backup_scheduler.lock"
STATUS_NAME = "status.json"
MAX_INTERVAL = 365 * 24 * 3600
MAX_TIMEOUT = 24 * 3600
MIN_INTERVAL = 0.01  # Smaller intervals cannot usefully schedule verified DB copies.


class SchedulerError(RuntimeError):
    pass


class SchedulerLocked(SchedulerError):
    pass


class StopRequest:
    def __init__(self):
        self.event = threading.Event()
        self.signal_number = None

    def request(self, signum=None, _frame=None):
        if self.signal_number is None:
            self.signal_number = signum
        self.event.set()

    def is_set(self):
        return self.event.is_set()

    def wait(self, seconds):
        return self.event.wait(seconds)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def validate_limits(interval, timeout, max_runs):
    for name, value, upper in (("interval", interval, MAX_INTERVAL), ("timeout", timeout, MAX_TIMEOUT)):
        if (type(value) not in {int, float} or value <= 0 or value > upper
                or not math.isfinite(value)):
            raise ValueError(f"{name} must be finite, positive and <= {upper} seconds")
    if interval < MIN_INTERVAL:
        raise ValueError(f"interval must be >= {MIN_INTERVAL} seconds")
    if max_runs is not None and (type(max_runs) is not int or max_runs <= 0):
        raise ValueError("max-runs must be a positive integer or omitted")


@contextmanager
def output_lock(output_dir):
    """Keep the inode: unlinking the lock file would allow split ownership."""
    if fcntl is None:
        raise SchedulerError("POSIX flock is required; Windows is not supported")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(output_dir / LOCK_NAME, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise SchedulerError("Lock path must be a regular file")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SchedulerLocked("Another scheduler holds the output directory lock") from exc
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def write_status(output_dir, status):
    """Atomic replacement of status only, never of a backup bundle."""
    payload = dict(status, updated_at_utc=utc_now())
    descriptor, temporary = tempfile.mkstemp(prefix=".status-", suffix=".tmp", dir=output_dir)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_dir / STATUS_NAME)
        database_backup.fsync_directory(output_dir)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    status["updated_at_utc"] = payload["updated_at_utc"]


def load_status(output_dir, source):
    path = output_dir / STATUS_NAME
    if not os.path.lexists(path):
        return {"format_version": FORMAT_VERSION, "source": str(source), "output_dir": str(output_dir),
                "counters": {"attempts": 0, "successes": 0, "failures": 0, "interrupted": 0},
                "last_attempt": None, "last_success": None, "last_error": None}
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "r", encoding="utf-8") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise SchedulerError("Status must be a regular file")
        previous = json.load(stream)
    if (not isinstance(previous, dict) or previous.get("format_version") != FORMAT_VERSION
            or previous.get("source") != str(source) or previous.get("output_dir") != str(output_dir)):
        raise SchedulerError("Existing status is incompatible; use a new output directory")
    counters = previous.get("counters", {})
    required = {"attempts", "successes", "failures", "interrupted"}
    if (not isinstance(counters, dict) or set(counters) != required
            or any(type(v) is not int or v < 0 for v in counters.values())):
        raise SchedulerError("Invalid persisted counters; status was not replaced")
    for key in ("last_attempt", "last_success", "last_error"):
        if key not in previous or (previous[key] is not None and not isinstance(previous[key], dict)):
            raise SchedulerError("Invalid persisted history; status was not replaced")
    attempt = previous["last_attempt"]
    pending = int(bool(attempt and attempt.get("status") == "in_progress"))
    if counters["attempts"] != counters["successes"] + counters["failures"] + counters["interrupted"] + pending:
        raise SchedulerError("Inconsistent persisted counters; status was not replaced")
    if pending:
        at = utc_now()
        previous["counters"]["interrupted"] += 1
        previous["last_attempt"] = dict(attempt, status="interrupted", finished_at_utc=at)
        previous["last_error"] = {"at_utc": at, "attempt_id": attempt.get("attempt_id"),
            "type": "InterruptedAttempt", "message": "Previous attempt ended without a persisted result; any published bundle is retained"}
    return previous


def next_deadline(previous, finished, interval):
    """First scheduled slot strictly after completion; never replay missed slots."""
    candidate = previous + interval
    skipped = 0
    if candidate <= finished:
        skipped = math.floor((finished - candidate) / interval) + 1
        candidate += skipped * interval
    # Floating point rounding must not produce a deadline in the past.
    if candidate <= finished:
        candidate = finished + interval
    return candidate, skipped


def run_scheduler(source, output_dir, *, interval=3600.0, timeout=60.0, max_runs=None, stop=None):
    """Run explicitly; preserve successful bundles/status across failed attempts.

    Status-write failures are fatal: do not continue creating unreported copies.
    SIGTERM/INT in the CLI request stop after the active cooperative backup ends.
    """
    validate_limits(interval, timeout, max_runs)
    source, output_dir = Path(source).resolve(), Path(output_dir).resolve()
    if source == output_dir or output_dir in source.parents:
        raise ValueError("Source must be outside the backup output directory")
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    stop = StopRequest() if stop is None else stop
    with output_lock(output_dir):
        status = load_status(output_dir, source)
        session_id = uuid4().hex
        status.update(session_id=session_id, pid=os.getpid(), state="starting", started_at_utc=utc_now(),
            interval_seconds=interval, timeout_seconds=timeout, max_runs=max_runs, session_attempts=0,
            session_successes=0, session_failures=0, skipped_slots=0, next_attempt_at_utc=None,
            stopped_at_utc=None, stop_reason=None)
        write_status(output_dir, status)
        deadline = time.monotonic()  # First attempt is immediate, even after a restart.
        while not stop.is_set():
            while not stop.is_set() and deadline > time.monotonic():
                stop.wait(min(60.0, max(0.0, deadline-time.monotonic())))
            if stop.is_set():
                break
            started = utc_now()
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            destination = output_dir / f"backup-{stamp}-{uuid4().hex}"
            attempt_id = f"{session_id}:{status['session_attempts']+1}"
            status["session_attempts"] += 1
            status["counters"]["attempts"] += 1
            status.update(state="backing_up", next_attempt_at_utc=None,
                last_attempt={"attempt_id": attempt_id, "destination": str(destination),
                              "status": "in_progress", "started_at_utc": started, "finished_at_utc": None})
            write_status(output_dir, status)
            try:
                result = database_backup.backup_database(source, destination, timeout=timeout)
            except Exception as exc:
                completed = utc_now()
                status["session_failures"] += 1
                status["counters"]["failures"] += 1
                status["last_error"] = {"at_utc": completed, "attempt_id": attempt_id,
                    "type": type(exc).__name__, "message": str(exc)}
                status["last_attempt"].update(status="failed", finished_at_utc=completed,
                                              error=dict(status["last_error"]))
            else:
                completed = utc_now()
                status["session_successes"] += 1
                status["counters"]["successes"] += 1
                status["last_success"] = {"attempt_id": attempt_id, "destination": str(destination),
                    "completed_at_utc": completed, "database_sha256": result["database_sha256"],
                    "manifest_sha256": result["manifest_sha256"], "elapsed_seconds": result["elapsed_seconds"],
                    "source_journal_mode": result["source_journal_mode"]}
                status["last_attempt"].update(status="succeeded", finished_at_utc=completed)
            deadline, skipped = next_deadline(deadline, time.monotonic(), interval)
            status["skipped_slots"] += skipped
            bounded_end = max_runs is not None and status["session_attempts"] >= max_runs
            if stop.is_set() or bounded_end:
                break
            status["state"] = "waiting"
            # Human-readable estimate only; wall time never schedules an attempt.
            status["next_attempt_at_utc"] = (datetime.now(timezone.utc) +
                timedelta(seconds=max(0.0, deadline-time.monotonic()))).isoformat()
            write_status(output_dir, status)
        reason = (f"signal:{signal.Signals(stop.signal_number).name}" if stop.signal_number else
                  "requested" if stop.is_set() else "max_runs")
        status.update(state="stopped", stopped_at_utc=utc_now(), stop_reason=reason, next_attempt_at_utc=None)
        write_status(output_dir, status)
        return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=3600.0, help="Seconds between monotonic scheduled slots")
    parser.add_argument("--timeout", type=float, default=60.0, help="Cooperative time limit passed to each backup")
    parser.add_argument("--max-runs", type=int, help="Number of attempts this session; omitted means until signal")
    args = parser.parse_args(argv)
    try:
        validate_limits(args.interval, args.timeout, args.max_runs)
    except ValueError as exc:
        parser.error(str(exc))
    stop = StopRequest()
    previous = {sig: signal.signal(sig, stop.request) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        status = run_scheduler(args.source, args.output_dir, interval=args.interval,
                               timeout=args.timeout, max_runs=args.max_runs, stop=stop)
    except SchedulerLocked as exc:
        print(json.dumps({"status": "locked", "error": str(exc)}), file=sys.stderr)
        return 3
    except (SchedulerError, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "type": type(exc).__name__, "error": str(exc)}), file=sys.stderr)
        return 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print(json.dumps(status, ensure_ascii=False, sort_keys=True))
    if stop.signal_number:
        return 128 + stop.signal_number
    return int(status["session_failures"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
