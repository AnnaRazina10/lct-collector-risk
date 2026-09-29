"""Two explicit archive modes, local draft tickets and dispatcher decision history."""
from __future__ import annotations
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from api import forecast_store
from src.serving import local_recommendations

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/app/risk_demo.json"
OBJECT_DATA = ROOT / "data/app/object_risk_demo.json"
DB = Path(os.environ.get("LCT_TICKET_DB", str(ROOT / "data/app/tickets.sqlite3")))
EntityMode = Literal["channel", "object"]
Decision = Literal["monitor", "inspect", "alarm_confirmed", "alarm_not_confirmed"]
Reason = Literal["needs_inspection", "planned_work", "sensor_or_connection", "journal_verified", "other"]
app = FastAPI(title="Коллектор — объектные и канальные архивные прогнозы", version="0.2.0")


def dataset(mode: EntityMode = "channel", run_id: str | None = None):
    if run_id is not None:
        try:
            data, metadata = forecast_store.load_run_with_metadata(DB, run_id, entity_mode=mode)
            data["archive_metadata"] = metadata
            return data
        except forecast_store.ForecastNotFoundError as error:
            raise HTTPException(404, "Выпуск не найден в выбранном режиме") from error
        except (forecast_store.ForecastIntegrityError, sqlite3.DatabaseError) as error:
            raise HTTPException(503, "Целостность архива выпусков не подтверждена") from error
    path = OBJECT_DATA if mode == "object" else DATA
    if not path.exists():
        raise HTTPException(503, f"Архивный набор режима {mode} ещё не сформирован")
    data = json.loads(path.read_text())
    data["entity_mode"] = mode
    if mode == "channel":
        data.setdefault("minimum_lead_hours", 0)
        data.setdefault("window_hours", 24)
        data.setdefault("issue_time", data.get("forecast_start"))
    return data


def public_card(card):
    # Archive facts are never part of the queue, ticket snapshot or feedback history.
    return {key: value for key, value in card.items() if not key.startswith("actual_") and key != "outcome"}


def find_card(risk_id: str, mode: EntityMode, run_id: str | None = None):
    data = dataset(mode, run_id)
    card = next((c for c in data["cards"] if c["id"] == risk_id), None)
    if card is None:
        raise HTTPException(404, "Карточка не найдена в выбранном режиме")
    return data, card


def recommendation(data, card):
    """Apply current review rules to server-resolved immutable object releases.

    Keep this separate from forecast cards and their original content hashes.
    Legacy JSON/channel modes do not have the required checked release context.
    """
    if data.get("entity_mode") != "object" or not data.get("run_id"):
        return None
    fields = local_recommendations.FACT_FIELDS | local_recommendations.OPTIONAL_FACT_FIELDS
    facts = {name: card[name] for name in fields if name in card}
    context = {name: data.get(name) for name in (
        "run_id", "feature_date", "feature_cutoff", "issue_time", "forecast_start", "forecast_end")}
    goal = {None: "any_alarm", "any_alarm": "any_alarm",
            forecast_store.ONSET_TARGET_KIND: "registered_episode_start_g1"}.get(data.get("target_kind"), "unsupported")
    context.update(card_id=card.get("id"), object_id=card.get("object_id"), mode="historical_replay", goal=goal)
    try:
        result = local_recommendations.recommend(facts, context)
    except local_recommendations.RecommendationValidationError:
        # Preserve readable old archives; do not infer advice from incomplete context.
        result = {"status": "unavailable", "steps": [], "requires_dispatcher_decision": True,
                  "external_send": False,
                  "message": "Для подробных рекомендаций недостаточно проверенного контекста выпуска."}
    return {**result, "generated_at": datetime.now(timezone.utc).isoformat(),
            "application_note": "Правила применены сейчас к архивным наблюдениям; историческое использование этих правил не подтверждается."}


def snapshot(data, card):
    fields = ["run_id", "mode", "entity_mode", "feature_date", "feature_cutoff", "issue_time", "forecast_start", "forecast_end",
              "minimum_lead_hours", "model", "model_sha256", "input_sha256", "threshold", "archive_metadata"]
    forecast = {k: data.get(k) for k in fields}
    forecast.update({k: data[k] for k in forecast_store.POLICY_META_FIELDS if k in data})
    result = {"forecast": forecast, "card": public_card(card)}
    advice = recommendation(data, card)
    if advice is not None:
        result["recommendation"] = advice
    return result


@contextmanager
def connection():
    DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        ensure_journal_schema(conn)
        with conn:
            yield conn
    finally:
        conn.close()


def ensure_journal_schema(conn):
    # Inspect every opened database, so replacement/restoration cannot leave a
    # stale process-level cache. Ready readers do not reserve the single writer.
    tickets = {row[1] for row in conn.execute("PRAGMA table_info(tickets)")}
    feedback = {row[1] for row in conn.execute("PRAGMA table_info(feedback)")}
    if {"id", "risk_id", "note", "created_at", "status", "entity_mode", "risk_snapshot"} <= tickets and {
        "id", "risk_id", "entity_mode", "decision", "reason", "operator", "note", "created_at", "risk_snapshot"
    } <= feedback:
        return
    # Serialize migrations only; re-read columns under this lock because another
    # first request may have completed the same migration after our initial read.
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("CREATE TABLE IF NOT EXISTS tickets (id TEXT PRIMARY KEY, risk_id TEXT UNIQUE, note TEXT, created_at TEXT, status TEXT)")
    columns = {row[1] for row in conn.execute("PRAGMA table_info(tickets)")}
    if "entity_mode" not in columns:
        conn.execute("ALTER TABLE tickets ADD COLUMN entity_mode TEXT NOT NULL DEFAULT 'channel'")
    if "risk_snapshot" not in columns:
        conn.execute("ALTER TABLE tickets ADD COLUMN risk_snapshot TEXT NOT NULL DEFAULT '{}'")
    conn.execute("""CREATE TABLE IF NOT EXISTS feedback (
        id TEXT PRIMARY KEY, risk_id TEXT NOT NULL, entity_mode TEXT NOT NULL,
        decision TEXT NOT NULL, reason TEXT NOT NULL, operator TEXT NOT NULL,
        note TEXT NOT NULL, created_at TEXT NOT NULL, risk_snapshot TEXT NOT NULL)""")
    conn.commit()


class TicketDraft(BaseModel):
    risk_id: str = Field(min_length=1, max_length=256)
    note: str = Field(default="", max_length=2000)
    entity_mode: EntityMode = "channel"
    run_id: str | None = Field(default=None, min_length=1, max_length=160)


class Feedback(BaseModel):
    risk_id: str = Field(min_length=1, max_length=256)
    entity_mode: EntityMode = "channel"
    run_id: str | None = Field(default=None, min_length=1, max_length=160)
    decision: Decision
    reason: Reason
    operator: str = Field(min_length=1, max_length=100)
    note: str = Field(default="", max_length=2000)

    @field_validator("operator")
    @classmethod
    def nonblank_operator(cls, value):
        if not value.strip():
            raise ValueError("Укажите имя диспетчера")
        return value.strip()


def saved_row(row):
    result = dict(row)
    result["risk_snapshot"] = json.loads(result.get("risk_snapshot") or "{}")
    return result


@app.get("/")
def index():
    return FileResponse(ROOT / "web/index.html")


@app.get("/api/health")
@app.get("/health", include_in_schema=False)
def health():
    modes = {"channel": DATA.exists(), "object": OBJECT_DATA.exists()}
    return {"status": "ok", "mode": "retrospective", "data_ready": any(modes.values()), "available_modes": modes}


@app.get("/api/risks")
@app.get("/risks", include_in_schema=False)
def risks(mode: EntityMode = "channel", run_id: str | None = Query(default=None, min_length=1, max_length=160)):
    data = dataset(mode, run_id)
    data["cards"] = [public_card(card) for card in data["cards"]]
    if mode == "object" and run_id is not None:
        data["recommendations"] = {card["id"]: recommendation(data, card) for card in data["cards"]}
    return data


@app.get("/api/forecast-runs")
def forecast_runs(mode: EntityMode | None = None, limit: int = Query(default=100, ge=1, le=500)):
    try:
        runs = forecast_store.list_runs(DB, entity_mode=mode, limit=limit)
    except (forecast_store.ForecastIntegrityError, sqlite3.DatabaseError) as error:
        raise HTTPException(503, "Целостность архива выпусков не подтверждена") from error
    return {"runs": runs, "limit": limit, "mode": "historical_replay",
            "note": "Сохранённые расчёты на исторических данных; текущий поток не подключён. created_at — время сохранения, issue_time — историческое время выпуска."}


@app.get("/api/risks/{risk_id}/outcome")
def outcome(risk_id: str, mode: EntityMode = "channel", run_id: str | None = Query(default=None, min_length=1, max_length=160)):
    data, card = find_card(risk_id, mode, run_id)
    if run_id is not None:
        raise HTTPException(404, "В сохранённых выпусках исторического расчёта исходы не хранятся")
    key = "actual_target_alarm" if mode == "object" else "actual_next_day_alarm"
    if key not in card:
        raise HTTPException(404, "Архивный факт для этого окна отсутствует")
    return {"risk_id": risk_id, "entity_mode": mode, key: card[key], "actual_alarm": card[key],
            "forecast_start": data.get("forecast_start"), "forecast_end": data.get("forecast_end"),
            "note": "Архивный факт целевого окна раскрыт отдельно; не использован в признаках прогноза. Регистрация тревоги не доказывает физическую поломку."}


@app.get("/api/risks/{risk_id}")
@app.get("/risks/{risk_id}", include_in_schema=False)
def risk(risk_id: str, mode: EntityMode = "channel", run_id: str | None = Query(default=None, min_length=1, max_length=160)):
    data, card = find_card(risk_id, mode, run_id)
    return snapshot(data, card)


@app.get("/api/tickets")
def tickets(mode: EntityMode | None = None):
    with connection() as conn:
        rows = conn.execute("SELECT * FROM tickets WHERE (? IS NULL OR entity_mode=?) ORDER BY created_at DESC", (mode, mode)).fetchall()
    return {"tickets": [saved_row(row) for row in rows], "external_submission": False}


@app.post("/api/tickets", status_code=201)
@app.post("/tickets/draft", status_code=201, include_in_schema=False)
def create_ticket(draft: TicketDraft):
    data, card = find_card(draft.risk_id, draft.entity_mode, draft.run_id)
    with connection() as conn:
        row = (str(uuid4()), draft.risk_id, draft.note, datetime.now(timezone.utc).isoformat(), "draft")
        result = conn.execute("""INSERT OR IGNORE INTO tickets
            (id,risk_id,note,created_at,status,entity_mode,risk_snapshot) VALUES (?,?,?,?,?,?,?)""",
            (*row, draft.entity_mode, json.dumps(snapshot(data, card), ensure_ascii=False)))
        existing = conn.execute("SELECT * FROM tickets WHERE risk_id=?", (draft.risk_id,)).fetchone()
        if existing["entity_mode"] != draft.entity_mode:
            raise HTTPException(409, "Идентификатор уже принадлежит другому режиму; сформируйте отдельную карточку")
        already = result.rowcount == 0
    return {**saved_row(existing), "external_submission": False, "already_exists": already}


@app.post("/api/feedback", status_code=201)
@app.post("/feedback", status_code=201, include_in_schema=False)
def create_feedback(feedback: Feedback):
    data, card = find_card(feedback.risk_id, feedback.entity_mode, feedback.run_id)
    row = (str(uuid4()), feedback.risk_id, feedback.entity_mode, feedback.decision, feedback.reason,
           feedback.operator, feedback.note, datetime.now(timezone.utc).isoformat(),
           json.dumps(snapshot(data, card), ensure_ascii=False))
    with connection() as conn:
        conn.execute("INSERT INTO feedback VALUES (?,?,?,?,?,?,?,?,?)", row)
        saved = conn.execute("SELECT * FROM feedback WHERE id=?", (row[0],)).fetchone()
    return {**saved_row(saved), "external_submission": False,
            "operator_identity_verified": False, "note_about_identity": "Имя указано локально; корпоративный вход не подключён."}


@app.get("/api/feedback")
def feedback_history(mode: EntityMode | None = None):
    with connection() as conn:
        rows = conn.execute("SELECT * FROM feedback WHERE (? IS NULL OR entity_mode=?) ORDER BY created_at DESC LIMIT 200", (mode, mode)).fetchall()
    return {"feedback": [saved_row(row) for row in rows], "external_submission": False}


@app.get("/api/journal")
def journal(mode: EntityMode | None = None):
    records = []
    with connection() as conn:
        for table, kind in [("tickets", "ticket_draft"), ("feedback", "dispatcher_decision")]:
            # table comes only from the fixed internal pair above.
            rows = conn.execute(f"SELECT * FROM {table} WHERE (? IS NULL OR entity_mode=?) ORDER BY created_at DESC LIMIT 200", (mode, mode)).fetchall()
            records.extend({**saved_row(row), "kind": kind} for row in rows)
    records.sort(key=lambda row: row["created_at"], reverse=True)
    return {"entries": records[:200], "limit": 200, "external_submission": False,
            "scope": "Локальные черновики и решения диспетчера; имя оператора не подтверждено корпоративным входом."}
