"""Local retrospective demo. Tickets are drafts stored on this computer only."""
from __future__ import annotations
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/app/risk_demo.json"
DB = Path(os.environ.get("LCT_TICKET_DB", str(ROOT / "data/app/tickets.sqlite3")))
app = FastAPI(title="Collector Risk — локальная демонстрация", version="0.1.0")


def dataset():
    if not DATA.exists():
        raise HTTPException(503, "Сначала сформируйте data/app/risk_demo.json")
    return json.loads(DATA.read_text())


@contextmanager
def connection():
    DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE IF NOT EXISTS tickets (id TEXT PRIMARY KEY, risk_id TEXT UNIQUE, note TEXT, created_at TEXT, status TEXT)")
    try:
        with conn:
            yield conn
    finally:
        conn.close()


class TicketDraft(BaseModel):
    risk_id: str = Field(min_length=1, max_length=100)
    note: str = Field(default="", max_length=2000)


@app.get("/")
def index():
    return FileResponse(ROOT / "web/index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "mode": "retrospective", "data_ready": DATA.exists()}


@app.get("/api/risks")
def risks():
    data = dataset()
    # Future outcomes are revealed only via a separate, explicit retrospective action.
    for card in data["cards"]:
        card.pop("actual_next_day_alarm", None)
    return data


@app.get("/api/risks/{risk_id}/outcome")
def outcome(risk_id: str):
    card = next((c for c in dataset()["cards"] if c["id"] == risk_id), None)
    if card is None:
        raise HTTPException(404, "Карточка не найдена")
    return {"risk_id": risk_id, "actual_next_day_alarm": card["actual_next_day_alarm"],
            "note": "Архивный факт следующего дня; не использован в признаках этого прогноза."}


@app.get("/api/tickets")
def tickets():
    with connection() as conn:
        rows = conn.execute("SELECT * FROM tickets ORDER BY created_at DESC").fetchall()
    return {"tickets": [dict(row) for row in rows], "external_submission": False}


@app.post("/api/tickets", status_code=201)
def create_ticket(draft: TicketDraft):
    if not any(c["id"] == draft.risk_id for c in dataset()["cards"]):
        raise HTTPException(404, "Карточка не найдена")
    with connection() as conn:
        existing = conn.execute("SELECT * FROM tickets WHERE risk_id=?", (draft.risk_id,)).fetchone()
        if existing:
            return {**dict(existing), "external_submission": False, "already_exists": True}
        row = (str(uuid4()), draft.risk_id, draft.note, datetime.now(timezone.utc).isoformat(), "draft")
        result = conn.execute("INSERT OR IGNORE INTO tickets VALUES (?,?,?,?,?)", row)
        if result.rowcount == 0:
            existing = conn.execute("SELECT * FROM tickets WHERE risk_id=?", (draft.risk_id,)).fetchone()
            return {**dict(existing), "external_submission": False, "already_exists": True}
    return dict(zip(["id", "risk_id", "note", "created_at", "status"], row)) | {"external_submission": False}
