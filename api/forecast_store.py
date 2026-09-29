"""Immutable, validated historical forecast releases; no model or outcome loading.

``created_at`` is the actual storage time. The payload's ``issue_time`` is the
historical simulation time and remains stable across idempotent publications.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

MSK = ZoneInfo("Europe/Moscow")
RUN_ID = re.compile(r"[A-Za-z0-9_.:-]{1,160}\Z")
SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
META_FIELDS = ("run_id", "entity_mode", "mode", "feature_date", "feature_cutoff",
               "issue_time", "forecast_start", "forecast_end", "minimum_lead_hours",
               "model_sha256", "input_sha256", "threshold")
POLICY_META_FIELDS = ("target_kind", "target_definition", "warning_policy", "score_kind")
ONSET_TARGET_KIND = "registered_episode_start_g1"
FUTURE_OUTCOME_FIELDS = {"outcome", "outcomes", "actual", "target_any_object_alarm_d_plus_2",
                         "onset_target", "alarm_target", "joint_target", "quiet_intermediate",
                         "records_intermediate", "records_target"}


class ForecastValidationError(ValueError):
    """Payload cannot represent a causal historical release."""


class ForecastConflictError(ValueError):
    """An immutable release or its card identifier already exists differently."""


class ForecastNotFoundError(LookupError):
    """Release does not exist in the requested entity mode."""


class ForecastIntegrityError(ValueError):
    """Stored bytes, metadata or schema no longer match their checksum."""


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ForecastValidationError(f"{name} must be a finite JSON number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ForecastValidationError(f"{name} must be finite")
    return value


def _timestamp(value, name):
    if not isinstance(value, str):
        raise ForecastValidationError(f"{name} must be an ISO timestamp with timezone")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ForecastValidationError(f"Invalid timestamp {name}") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ForecastValidationError(f"{name} must include a timezone")
    return parsed


def _check_json(value, path="payload"):
    """Check every nested field, not only the top-level card label."""
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ForecastValidationError(f"Non-string key at {path}")
            if key.lower().startswith("actual_") or key.lower() in FUTURE_OUTCOME_FIELDS:
                raise ForecastValidationError(f"Future outcome field forbidden: {path}.{key}")
            _check_json(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_json(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ForecastValidationError(f"Nonfinite number at {path}")
    elif value is not None and not isinstance(value, (str, bool, int, float)):
        raise ForecastValidationError(f"Non-JSON value at {path}")


def _warning_policy(payload):
    """Recognize only the frozen onset policy; absence preserves legacy behavior."""
    if "warning_policy" not in payload:
        if payload.get("target_kind") == ONSET_TARGET_KIND:
            raise ForecastValidationError("Registered onset releases require an explicit warning_policy")
        return None
    policy = payload["warning_policy"]
    if not isinstance(policy, dict) or set(policy) != {"kind", "k", "tie_break"}:
        raise ForecastValidationError("warning_policy must contain exactly kind, k and tie_break")
    if policy["kind"] != "daily_top_k" or policy["tie_break"] != "score_desc_object_id_asc":
        raise ForecastValidationError("Unsupported warning policy or tie break")
    if isinstance(policy["k"], bool) or not isinstance(policy["k"], int) or policy["k"] != 10:
        raise ForecastValidationError("The frozen onset warning policy requires integer k=10")
    if payload.get("target_kind") != ONSET_TARGET_KIND:
        raise ForecastValidationError("daily_top_k requires target_kind registered_episode_start_g1")
    if not isinstance(payload.get("target_definition"), str) or not payload["target_definition"].strip():
        raise ForecastValidationError("Onset target_definition must be a nonempty string")
    if "score_kind" in payload and (not isinstance(payload["score_kind"], str) or not payload["score_kind"].strip()):
        raise ForecastValidationError("score_kind must be a nonempty string when provided")
    return policy


def validate_payload(payload):
    if not isinstance(payload, dict):
        raise ForecastValidationError("payload must be an object")
    _check_json(payload)
    missing = set(META_FIELDS).difference(payload)
    if missing:
        raise ForecastValidationError(f"Missing fields: {', '.join(sorted(missing))}")
    run_id = payload["run_id"]
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
        raise ForecastValidationError("Invalid run_id")
    if payload["entity_mode"] != "object" or payload["mode"] != "historical_replay":
        raise ForecastValidationError("Only object historical_replay releases are supported")
    feature_date = payload["feature_date"]
    try:
        feature_day = date.fromisoformat(feature_date)
    except (TypeError, ValueError) as error:
        raise ForecastValidationError("feature_date must be YYYY-MM-DD") from error
    if feature_day.isoformat() != feature_date:
        raise ForecastValidationError("feature_date must be YYYY-MM-DD")
    cutoff = _timestamp(payload["feature_cutoff"], "feature_cutoff")
    issue = _timestamp(payload["issue_time"], "issue_time")
    start = _timestamp(payload["forecast_start"], "forecast_start")
    end = _timestamp(payload["forecast_end"], "forecast_end")
    feature_day_end = datetime.combine(feature_day + timedelta(days=1), time(), tzinfo=MSK)
    if cutoff.astimezone(MSK).date() != feature_day or cutoff >= issue or issue < feature_day_end:
        raise ForecastValidationError("Feature cutoff/day must precede issue_time; a full daily feature must be available")
    lead = (start-issue).total_seconds()/3600
    declared_lead = _number(payload["minimum_lead_hours"], "minimum_lead_hours")
    if lead < 24 or not math.isclose(declared_lead, lead, rel_tol=0, abs_tol=1e-9):
        raise ForecastValidationError("Forecast requires >=24 hours from issue to window and a matching minimum_lead_hours")
    if end <= start:
        raise ForecastValidationError("Forecast window must have positive duration")
    if "window_hours" in payload and not math.isclose(_number(payload["window_hours"], "window_hours"), (end-start).total_seconds()/3600, rel_tol=0, abs_tol=1e-9):
        raise ForecastValidationError("window_hours does not match timestamps")
    for field in ("model_sha256", "input_sha256"):
        if not isinstance(payload[field], str) or not SHA256.fullmatch(payload[field]):
            raise ForecastValidationError(f"{field} must be a SHA-256 hex string")
    threshold = _number(payload["threshold"], "threshold")
    if not 0 <= threshold <= 1:
        raise ForecastValidationError("threshold must be within [0,1]")
    policy = _warning_policy(payload)
    cards = payload.get("cards")
    if not isinstance(cards, list) or not cards:
        raise ForecastValidationError("cards must be a nonempty list")
    ids, object_ids = set(), set()
    for card in cards:
        if not isinstance(card, dict):
            raise ForecastValidationError("Each card must be an object")
        cid = card.get("id")
        if not isinstance(cid, str) or not 1 <= len(cid) <= 256 or run_id not in cid or cid in ids:
            raise ForecastValidationError("Card ids must be unique, <=256 characters and contain run_id")
        ids.add(cid)
        if card.get("entity_mode", "object") != "object":
            raise ForecastValidationError("Card entity_mode mismatch")
        if policy is not None:
            oid = card.get("object_id")
            if isinstance(oid, bool) or not isinstance(oid, (str, int)) or not str(oid).strip():
                raise ForecastValidationError("Every onset card requires a nonempty string or integer object_id")
            rank = card.get("rank")
            if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= len(cards):
                raise ForecastValidationError("Every onset card requires an integer rank in 1..N")
        if "object_id" in card:
            oid = str(card["object_id"])
            if oid in object_ids:
                raise ForecastValidationError("Duplicate object_id in release")
            object_ids.add(oid)
        score = _number(card.get("score"), "card.score")
        if not 0 <= score <= 1:
            raise ForecastValidationError("Card score must be within [0,1]")
        if not isinstance(card.get("warning"), bool):
            raise ForecastValidationError("Card warning must be boolean")
        if policy is None and card["warning"] != (score >= threshold):
            raise ForecastValidationError("Legacy card warning must equal score >= threshold")
    if policy is not None:
        k = policy["k"]
        if len(cards) < k:
            raise ForecastValidationError("Onset daily top-k requires at least k objects")
        ordered = sorted(cards, key=lambda card: (-card["score"], str(card["object_id"])))
        for expected_rank, card in enumerate(ordered, start=1):
            if card["rank"] != expected_rank or card["warning"] != (expected_rank <= k):
                raise ForecastValidationError("Onset rank and warning must follow score descending then string object_id ascending")
        if threshold != ordered[k-1]["score"]:
            raise ForecastValidationError("Onset threshold must equal the boundary kth card score; ranks resolve ties")
        if sum(card["warning"] for card in cards) != k:
            raise ForecastValidationError("Onset warning count must equal k")
    for field, expected in (("total_objects", len(cards)), ("shown_cards", len(cards)),
                            ("warnings_count", sum(card["warning"] for card in cards))):
        if field in payload and (isinstance(payload[field], bool) or not isinstance(payload[field], int) or payload[field] != expected):
            raise ForecastValidationError(f"{field} does not match cards")
    return payload


def canonical_payload(payload):
    validate_payload(payload)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _initialize(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS forecast_runs (
        run_id TEXT PRIMARY KEY, entity_mode TEXT NOT NULL, mode TEXT NOT NULL,
        issue_epoch REAL NOT NULL, metadata_json TEXT NOT NULL,
        payload_json TEXT NOT NULL, content_sha256 TEXT NOT NULL, created_at TEXT NOT NULL)""")
    conn.execute("CREATE INDEX IF NOT EXISTS forecast_runs_issue ON forecast_runs (issue_epoch DESC, run_id)")
    conn.execute("CREATE TABLE IF NOT EXISTS forecast_card_ids (card_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES forecast_runs(run_id))")
    for table in ("forecast_runs", "forecast_card_ids"):
        for operation in ("UPDATE", "DELETE"):
            conn.execute(f"CREATE TRIGGER IF NOT EXISTS {table}_immutable_{operation.lower()} BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT, 'Forecast releases are immutable'); END")


def _metadata(payload, digest, created_at):
    metadata = {**{field: payload[field] for field in META_FIELDS}, "cards_count": len(payload["cards"]),
                "content_sha256": digest, "created_at": created_at}
    # Existing releases already carry score_kind in their payload but not in
    # metadata_json. Preserve their exact metadata schema, digest and retry value.
    if "warning_policy" in payload:
        metadata.update({field: payload[field] for field in POLICY_META_FIELDS if field in payload})
    return metadata


def _verified_row(row):
    try:
        raw = row["payload_json"]
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if digest != row["content_sha256"]:
            raise ForecastIntegrityError("Forecast content checksum mismatch")
        payload = json.loads(raw)
        if canonical_payload(payload) != raw:
            raise ForecastIntegrityError("Forecast payload is not canonical")
        metadata = _metadata(payload, digest, row["created_at"])
        if json.loads(row["metadata_json"]) != metadata:
            raise ForecastIntegrityError("Forecast metadata mismatch")
        if (payload["run_id"] != row["run_id"] or payload["entity_mode"] != row["entity_mode"]
                or payload["mode"] != row["mode"] or _timestamp(payload["issue_time"], "issue_time").timestamp() != row["issue_epoch"]):
            raise ForecastIntegrityError("Forecast index mismatch")
        _timestamp(row["created_at"], "created_at")
        return payload, metadata
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, ForecastIntegrityError):
            raise
        raise ForecastIntegrityError(f"Invalid stored forecast: {error}") from error


def publish_run(database: Path, payload: dict) -> dict:
    """Atomically publish once; equal canonical content is an idempotent retry."""
    raw = canonical_payload(payload)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    database = Path(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(database, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("BEGIN IMMEDIATE")
            _initialize(conn)
            existing = conn.execute("SELECT * FROM forecast_runs WHERE run_id=?", (payload["run_id"],)).fetchone()
            if existing is not None:
                _, metadata = _verified_row(existing)
                if raw != existing["payload_json"]:
                    raise ForecastConflictError("run_id already exists with different content")
                return metadata
            created_at = datetime.now(timezone.utc).isoformat()
            metadata = _metadata(payload, digest, created_at)
            conn.execute("INSERT INTO forecast_runs VALUES (?,?,?,?,?,?,?,?)",
                         (payload["run_id"], payload["entity_mode"], payload["mode"],
                          _timestamp(payload["issue_time"], "issue_time").timestamp(),
                          json.dumps(metadata, ensure_ascii=False, sort_keys=True), raw, digest, created_at))
            try:
                conn.executemany("INSERT INTO forecast_card_ids VALUES (?,?)", [(card["id"], payload["run_id"]) for card in payload["cards"]])
            except sqlite3.IntegrityError as error:
                raise ForecastConflictError("A card id already belongs to another release") from error
        return metadata
    finally:
        conn.close()


def _read_rows(database, run_id=None, entity_mode=None, limit=100):
    database = Path(database)
    if not database.exists():
        return []
    conn = sqlite3.connect(database.resolve().as_uri()+"?mode=ro", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        if conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='forecast_runs'").fetchone() is None:
            return []
        return conn.execute("""SELECT * FROM forecast_runs
            WHERE (? IS NULL OR run_id=?) AND (? IS NULL OR entity_mode=?)
            ORDER BY issue_epoch DESC, run_id DESC LIMIT ?""", (run_id, run_id, entity_mode, entity_mode, limit)).fetchall()
    finally:
        conn.close()


def _read_one(database, run_id, entity_mode=None):
    rows = _read_rows(database, run_id=run_id, entity_mode=entity_mode, limit=1)
    if not rows:
        raise ForecastNotFoundError("Forecast release not found in the selected mode")
    return _verified_row(rows[0])


def load_run(database: Path, run_id: str, entity_mode=None) -> dict:
    return _read_one(database, run_id, entity_mode)[0]


def load_run_with_metadata(database: Path, run_id: str, entity_mode=None) -> tuple[dict, dict]:
    """Return payload and metadata from one checked row and read snapshot."""
    return _read_one(database, run_id, entity_mode)


def get_run_metadata(database: Path, run_id: str, entity_mode=None) -> dict:
    return _read_one(database, run_id, entity_mode)[1]


def list_runs(database: Path, entity_mode=None, limit=100) -> list[dict]:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
        raise ValueError("limit must be an integer between 1 and 500")
    return [_verified_row(row)[1] for row in _read_rows(database, entity_mode=entity_mode, limit=limit)]
