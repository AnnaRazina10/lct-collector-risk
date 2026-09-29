"""Regression checks for mode isolation, future-label redaction and local history."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from api import main


class ObjectApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        folder = Path(self.tmp.name)
        self.channel_path = folder / "channel.json"
        self.object_path = folder / "object.json"
        self.db = folder / "decisions.sqlite3"
        self.channel_path.write_text(json.dumps({"cards": [{"id": "channel_1", "score": .2, "actual_next_day_alarm": True}]}))
        self.object_data = {"feature_date": "2026-06-28", "issue_time": "2026-06-29T00:00:00+03:00",
            "forecast_start": "2026-06-30T00:00:00+03:00", "forecast_end": "2026-07-01T00:00:00+03:00",
            "minimum_lead_hours": 24, "model": "frozen", "threshold": .3345,
            "cards": [{"id": "object_1", "object_name": "Объект 1", "score": .6,
                       "actual_target_alarm": True, "actual_hidden": 9, "outcome": True}]}
        self.object_path.write_text(json.dumps(self.object_data))
        self.patch = patch.multiple(main, DATA=self.channel_path, OBJECT_DATA=self.object_path, DB=self.db)
        self.patch.start()
        self.client = TestClient(main.app)

    def tearDown(self):
        self.client.close()
        self.patch.stop()
        self.tmp.cleanup()

    def decision(self, **changes):
        body = {"risk_id": "object_1", "entity_mode": "object", "decision": "inspect",
                "reason": "needs_inspection", "operator": "  Смена 1  ", "note": "Проверка"}
        body.update(changes)
        return self.client.post("/api/feedback", json=body)

    def assert_no_outcome(self, value):
        if isinstance(value, dict):
            self.assertFalse(any(k.startswith("actual_") or k == "outcome" for k in value))
            for item in value.values():
                self.assert_no_outcome(item)
        elif isinstance(value, list):
            for item in value:
                self.assert_no_outcome(item)

    def test_mode_and_outcome_isolation(self):
        legacy = self.client.get("/api/risks").json()
        self.assertEqual(legacy["entity_mode"], "channel")
        obj = self.client.get("/api/risks?mode=object").json()
        self.assertEqual(obj["minimum_lead_hours"], 24)
        self.assert_no_outcome(obj)
        detail = self.client.get("/api/risks/object_1?mode=object").json()
        self.assert_no_outcome(detail)
        self.assertEqual(detail["forecast"]["issue_time"], self.object_data["issue_time"])
        self.assertEqual(self.client.get("/api/risks/object_1").status_code, 404)
        self.assertEqual(self.client.get("/api/risks/channel_1?mode=object").status_code, 404)
        self.assertEqual(self.client.get("/api/risks?mode=invalid").status_code, 422)
        self.assertTrue(self.client.get("/api/risks/object_1/outcome?mode=object").json()["actual_alarm"])

    def test_draft_and_decision_snapshots_survive_dataset_changes(self):
        draft = {"risk_id": "object_1", "entity_mode": "object", "note": "Исходная заметка"}
        first = self.client.post("/api/tickets", json=draft)
        self.assertEqual(first.status_code, 201)
        self.assert_no_outcome(first.json())
        self.assertEqual(self.decision().status_code, 201)
        self.object_data["cards"][0]["score"] = .1
        self.object_path.write_text(json.dumps(self.object_data))
        again = self.client.post("/api/tickets", json={**draft, "note": "Перезапись"}).json()
        self.assertTrue(again["already_exists"])
        self.assertEqual(again["id"], first.json()["id"])
        self.assertEqual(again["note"], "Исходная заметка")
        self.assertEqual(again["risk_snapshot"]["card"]["score"], .6)
        self.assertEqual(self.decision(decision="monitor").status_code, 201)
        entries = self.client.get("/api/journal?mode=object").json()["entries"]
        self.assertEqual(len(entries), 3)
        self.assert_no_outcome(entries)
        decisions = [e for e in entries if e["kind"] == "dispatcher_decision"]
        self.assertEqual({e["decision"] for e in decisions}, {"monitor", "inspect"})
        self.assertEqual({e["risk_snapshot"]["card"]["score"] for e in decisions}, {.1, .6})
        self.assertTrue(all(e["operator"] == "Смена 1" for e in decisions))
        self.assertEqual(self.client.get("/api/journal?mode=channel").json()["entries"], [])
        self.assertEqual(len(self.client.get("/api/feedback?mode=object").json()["feedback"]), 2)
        self.assertFalse(self.client.get("/api/tickets?mode=object").json()["external_submission"])

    def test_feedback_validation(self):
        for changes in [{"operator": "  "}, {"reason": "unknown"}, {"decision": "unknown"},
                        {"entity_mode": "unknown"}, {"note": "x" * 2001}]:
            with self.subTest(changes=changes):
                self.assertEqual(self.decision(**changes).status_code, 422)
        self.assertEqual(self.decision(risk_id="channel_1").status_code, 404)
        result = self.decision().json()
        self.assertFalse(result["operator_identity_verified"])
        self.assertFalse(result["external_submission"])

    def test_legacy_database_migrates_without_losing_drafts(self):
        with sqlite3.connect(self.db) as db:
            db.execute("CREATE TABLE tickets (id TEXT PRIMARY KEY, risk_id TEXT UNIQUE, note TEXT, created_at TEXT, status TEXT)")
            db.execute("INSERT INTO tickets VALUES (?,?,?,?,?)", ("old", "channel_1", "Старый черновик", "2026-01-01T00:00:00+00:00", "draft"))
        rows = self.client.get("/api/tickets?mode=channel").json()["tickets"]
        self.assertEqual(rows[0]["note"], "Старый черновик")
        self.assertEqual(rows[0]["entity_mode"], "channel")
        self.assertEqual(rows[0]["risk_snapshot"], {})
        self.assertEqual(self.decision().status_code, 201)

    def test_same_identifier_cannot_reuse_other_mode_draft(self):
        self.client.post("/api/tickets", json={"risk_id": "channel_1"})
        self.object_data["cards"][0]["id"] = "channel_1"
        self.object_path.write_text(json.dumps(self.object_data))
        collision = self.client.post("/api/tickets", json={"risk_id": "channel_1", "entity_mode": "object"})
        self.assertEqual(collision.status_code, 409)
        self.assertEqual(len(self.client.get("/api/tickets").json()["tickets"]), 1)


if __name__ == "__main__":
    unittest.main()
