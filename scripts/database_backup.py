#!/usr/bin/env python3
"""SQLite online backup/verified restore; no overwrite and no API dependency."""
from __future__ import annotations

import argparse
import base64
from contextlib import closing
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import time

FORMAT_VERSION = 1
DB_NAME = "database.sqlite3"
MANIFEST_NAME = "manifest.json"
SIDECARS = ("-wal", "-shm", "-journal")


class BackupError(RuntimeError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def quote_identifier(value):
    return '"' + value.replace('"', '""') + '"'


def typed_value(value):
    if value is None:
        return ["null"]
    if isinstance(value, bytes):
        return ["blob", base64.b64encode(value).decode("ascii")]
    if isinstance(value, float):
        return ["real", value.hex()]
    if isinstance(value, int):
        return ["integer", str(value)]
    return ["text", value]


def check_deadline(deadline):
    if time.monotonic() > deadline:
        raise TimeoutError("Backup/restore exceeded its time limit")


def connect_readonly(path, deadline, *, immutable=False):
    path = Path(path).absolute()
    if not path.is_file():
        raise FileNotFoundError(path)
    # immutable is ONLY used for an already sealed standalone bundle, never a live WAL source.
    uri = path.as_uri() + "?mode=ro" + ("&immutable=1" if immutable else "")
    conn = sqlite3.connect(uri, uri=True, timeout=5)
    conn.execute("PRAGMA query_only=ON")
    conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
    return conn


def inventory(conn, deadline):
    """Verify integrity; hash schema and every typed row, independent of row order.

    Row digests are sorted per table (32 bytes/row plus Python overhead). Internal
    tables such as sqlite_sequence are included separately from application tables.
    """
    check_deadline(deadline)
    integrity = [row[0] for row in conn.execute("PRAGMA integrity_check")]
    if integrity != ["ok"]:
        raise BackupError(f"SQLite integrity_check failed: {integrity[:5]}")
    schema = [list(row) for row in conn.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_schema ORDER BY type,name,tbl_name")]
    tables = {}
    for (name,) in conn.execute("SELECT name FROM sqlite_schema WHERE type='table' ORDER BY name").fetchall():
        check_deadline(deadline)
        columns = [list(row) for row in conn.execute(f"PRAGMA table_xinfo({quote_identifier(name)})")]
        cursor = conn.execute(f"SELECT * FROM {quote_identifier(name)}")
        row_digests = []
        for number, row in enumerate(cursor):
            if number % 1024 == 0:
                check_deadline(deadline)
            row_digests.append(hashlib.sha256(canonical([typed_value(value) for value in row])).digest())
        digest = hashlib.sha256()
        for value in sorted(row_digests):
            digest.update(value)
        tables[name] = {"row_count": len(row_digests), "columns": columns,
                        "payload_sha256": digest.hexdigest(), "internal": name.startswith("sqlite_")}
    return {"integrity_check": "ok", "schema": schema,
            "schema_sha256": hashlib.sha256(canonical(schema)).hexdigest(), "tables": tables,
            "user_tables": [name for name in tables if not name.startswith("sqlite_")],
            "user_version": conn.execute("PRAGMA user_version").fetchone()[0],
            "application_id": conn.execute("PRAGMA application_id").fetchone()[0],
            "encoding": conn.execute("PRAGMA encoding").fetchone()[0]}


def require_new(path, *, database=False):
    if os.path.lexists(path):
        raise FileExistsError(f"Refusing to overwrite an existing path: {path}")
    if database:
        for suffix in SIDECARS:
            sidecar = Path(str(path) + suffix)
            if os.path.lexists(sidecar):
                raise FileExistsError(f"Refusing destination with an existing SQLite sidecar: {sidecar}")


def atomic_rename_new(source, destination):
    """Atomic rename with OS-level no-replace, including a concurrent destination creator.

    Plain os.rename/os.replace on POSIX can overwrite after exists() checks.
    Fail closed if the platform/filesystem does not support exclusive renaming.
    """
    source, destination = os.fsencode(source), os.fsencode(destination)
    if sys.platform == "win32":
        os.rename(source, destination)  # Windows rename fails if destination exists.
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        function = libc.renamex_np
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        arguments = (source, destination, 0x00000004)  # RENAME_EXCL, Apple sys/stdio.h
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        function = libc.renameat2
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        arguments = (-100, source, -100, destination, 1)  # AT_FDCWD, RENAME_NOREPLACE
    else:
        raise BackupError("Atomic no-replace rename is unavailable on this platform")
    function.restype = ctypes.c_int
    if function(*arguments) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), os.fsdecode(destination))


def fsync_file(path):
    with Path(path).open("rb") as stream:
        os.fsync(stream.fileno())


def fsync_directory(path):
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def copy_online(source, destination, deadline):
    def progress(status, remaining, total):
        check_deadline(deadline)

    source.backup(destination, pages=128, progress=progress, sleep=.05)
    # Only the newly-created destination is changed to a standalone rollback DB.
    if destination.execute("PRAGMA journal_mode=DELETE").fetchone()[0] != "delete":
        raise BackupError("Cannot produce a standalone backup database")


def backup_database(source, destination, *, timeout=60.0):
    started = time.monotonic()
    deadline = started + timeout
    source, destination = Path(source).absolute(), Path(destination).absolute()
    require_new(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".lct-backup-", dir=destination.parent))
    try:
        copied = temporary / DB_NAME
        with closing(connect_readonly(source, deadline)) as live:
            live.execute("BEGIN")  # Pin a consistent read snapshot for inventory AND backup.
            live.execute("SELECT count(*) FROM sqlite_schema").fetchone()
            mode = live.execute("PRAGMA journal_mode").fetchone()[0]
            original = inventory(live, deadline)
            snapshot_at = utc_now()
            with closing(sqlite3.connect(copied)) as output:
                copy_online(live, output, deadline)
                checked = inventory(output, deadline)
            if original != checked:
                raise BackupError("Backup schema, row counts or typed payload differs from the source snapshot")
            live.rollback()
        copied.chmod(0o600)
        manifest = {"format_version": FORMAT_VERSION, "created_at_utc": utc_now(),
                    "snapshot_recorded_at_utc": snapshot_at, "sqlite_version": sqlite3.sqlite_version,
                    "method": "sqlite3.Connection.backup with pinned source read transaction",
                    "source_path": str(source), "source_journal_mode": mode,
                    "database_file": DB_NAME, "database_bytes": copied.stat().st_size,
                    "database_sha256": sha256(copied), "inventory": checked,
                    "source_snapshot_matches_backup": True,
                    "scope": "main database only; external JSON/model files and attached databases excluded"}
        manifest_path = temporary / MANIFEST_NAME
        manifest_path.write_bytes(canonical(manifest) + b"\n")
        manifest_path.chmod(0o600)
        fsync_file(copied)
        fsync_file(manifest_path)
        fsync_directory(temporary)
        check_deadline(deadline)
        atomic_rename_new(temporary, destination)
        fsync_directory(destination.parent)
        return {"passed": True, "operation": "backup", "destination": str(destination),
                "elapsed_seconds": time.monotonic() - started, "database_sha256": manifest["database_sha256"],
                "manifest_sha256": sha256(destination / MANIFEST_NAME),
                "source_journal_mode": mode, "inventory": checked}
    finally:
        # Only our unique staging directory is removed, never source/final destination.
        if temporary.exists():
            shutil.rmtree(temporary)


def bundle_manifest(bundle):
    bundle = Path(bundle).absolute()
    manifest = json.loads((bundle / MANIFEST_NAME).read_text())
    if not isinstance(manifest, dict) or manifest.get("format_version") != FORMAT_VERSION or manifest.get("database_file") != DB_NAME:
        raise BackupError("Unsupported or malformed backup manifest")
    database = bundle / DB_NAME
    if database.is_symlink() or not database.is_file():
        raise BackupError("Backup database must be a regular non-symlink file")
    for suffix in SIDECARS:
        if os.path.lexists(str(database) + suffix):
            raise BackupError("A sealed backup must not have SQLite WAL/journal sidecars")
    if database.stat().st_size != manifest["database_bytes"] or sha256(database) != manifest["database_sha256"]:
        raise BackupError("Backup database checksum/size differs from its manifest")
    return database, manifest


def verify_backup(bundle, *, timeout=60.0):
    deadline = time.monotonic() + timeout
    database, manifest = bundle_manifest(bundle)
    with closing(connect_readonly(database, deadline, immutable=True)) as source:
        checked = inventory(source, deadline)
    if checked != manifest["inventory"] or sha256(database) != manifest["database_sha256"]:
        raise BackupError("Backup integrity/schema/rows/payload verification failed")
    return {"passed": True, "operation": "verify", "database_sha256": manifest["database_sha256"],
            "manifest_sha256": sha256(Path(bundle) / MANIFEST_NAME), "inventory": checked}


def restore_database(bundle, destination, *, timeout=60.0):
    started = time.monotonic()
    deadline = started + timeout
    destination = Path(destination).absolute()
    require_new(destination, database=True)
    database, manifest = bundle_manifest(bundle)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".lct-restore-", suffix=".sqlite3", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with closing(connect_readonly(database, deadline, immutable=True)) as source:
            source.execute("BEGIN")
            original = inventory(source, deadline)
            if original != manifest["inventory"]:
                raise BackupError("Backup schema/rows/payload differs from its manifest")
            with closing(sqlite3.connect(temporary)) as restored:
                copy_online(source, restored, deadline)
                checked = inventory(restored, deadline)
            if checked != original:
                raise BackupError("Restored schema, row counts or typed payload differs from the backup")
            if sha256(database) != manifest["database_sha256"]:
                raise BackupError("Backup file changed during restoration")
        fsync_file(temporary)
        restored_sha = sha256(temporary)
        check_deadline(deadline)
        require_new(destination, database=True)
        atomic_rename_new(temporary, destination)
        fsync_directory(destination.parent)
        return {"passed": True, "operation": "restore", "destination": str(destination),
                "elapsed_seconds": time.monotonic() - started,
                "backup_database_sha256": manifest["database_sha256"], "restored_sha256": restored_sha,
                "manifest_sha256": sha256(Path(bundle) / MANIFEST_NAME),
                "inventory_matches_backup": True, "inventory": checked}
    finally:
        for path in [temporary, *(Path(str(temporary) + suffix) for suffix in SIDECARS)]:
            if path.exists():
                path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    backup = subparsers.add_parser("backup", help="Create a new bundle directory from a live SQLite database")
    backup.add_argument("--source", type=Path, required=True)
    backup.add_argument("--destination", type=Path, required=True)
    restore = subparsers.add_parser("restore", help="Restore a verified bundle to a NEW database path")
    restore.add_argument("--bundle", type=Path, required=True)
    restore.add_argument("--destination", type=Path, required=True)
    verify = subparsers.add_parser("verify", help="Verify a standalone backup bundle without changing it")
    verify.add_argument("--bundle", type=Path, required=True)
    for command in (backup, restore, verify):
        command.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    try:
        if args.operation == "backup":
            result = backup_database(args.source, args.destination, timeout=args.timeout)
        elif args.operation == "restore":
            result = restore_database(args.bundle, args.destination, timeout=args.timeout)
        else:
            result = verify_backup(args.bundle, timeout=args.timeout)
    except (OSError, sqlite3.Error, BackupError, ValueError, KeyError, TimeoutError) as exc:
        print(json.dumps({"passed": False, "operation": args.operation, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
