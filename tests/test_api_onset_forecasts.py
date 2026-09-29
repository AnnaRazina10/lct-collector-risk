"""Onset target/policy/rank are preserved in API and dispatcher history."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from api import forecast_store, main
from tests.test_forecast_onset_policy import onset_payload
from tests.test_forecast_store import payload


class OnsetForecastApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); folder = Path(self.tmp.name)
        self.db = folder / "app.sqlite3"
        self.legacy = payload("any_alarm"); self.onset = onset_payload("registered_start")
        forecast_store.publish_run(self.db, self.legacy)
        forecast_store.publish_run(self.db, self.onset)
        old_json = folder / "legacy.json"
        old_json.write_text(json.dumps({"cards": [{"id": "old", "score": .3, "actual_target_alarm": True}]}))
        self.patch = patch.multiple(main, DB=self.db, DATA=old_json, OBJECT_DATA=old_json)
        self.patch.start(); self.client = TestClient(main.app)

    def tearDown(self):
        self.client.close(); self.patch.stop(); self.tmp.cleanup()

    def test_target_metadata_and_detail_coexist_with_legacy(self):
        meta = {r["run_id"]: r for r in self.client.get("/api/forecast-runs?mode=object").json()["runs"]}
        self.assertNotIn("target_kind", meta["any_alarm"])
        for name in ("target_kind", "target_definition", "warning_policy", "score_kind"):
            self.assertEqual(meta["registered_start"][name], self.onset[name])
        release = self.client.get("/api/risks?mode=object&run_id=registered_start").json()
        self.assertEqual(sum(c["warning"] for c in release["cards"]), 10)
        card = release["cards"][0]
        detail = self.client.get(f"/api/risks/{card['id']}?mode=object&run_id=registered_start").json()
        self.assertEqual(detail["card"]["rank"], card["rank"])
        self.assertEqual(detail["forecast"]["warning_policy"], self.onset["warning_policy"])
        self.assertEqual(self.client.get(f"/api/risks/{card['id']}/outcome?mode=object&run_id=registered_start").status_code, 404)
        self.assertTrue(self.client.get("/api/risks/old/outcome?mode=object").json()["actual_alarm"])

    def test_ticket_and_feedback_snapshot_preserve_onset_semantics_in_journal(self):
        card = self.onset["cards"][0]
        identity = {"risk_id": card["id"], "entity_mode": "object", "run_id": self.onset["run_id"]}
        ticket = self.client.post("/api/tickets", json=identity)
        feedback = self.client.post("/api/feedback", json={**identity, "operator": "Демо",
                                   "decision": "inspect", "reason": "needs_inspection"})
        self.assertEqual(ticket.status_code, 201); self.assertEqual(feedback.status_code, 201)
        entries = self.client.get("/api/journal?mode=object").json()["entries"]
        self.assertEqual(len(entries), 2)
        for entry in entries:
            snap = entry["risk_snapshot"]
            for name in ("target_kind", "target_definition", "warning_policy", "score_kind"):
                self.assertEqual(snap["forecast"][name], self.onset[name])
            self.assertEqual(snap["card"]["rank"], card["rank"])
            self.assertEqual(snap["forecast"]["run_id"], "registered_start")
        wrong = {**identity, "run_id": "any_alarm"}
        self.assertEqual(self.client.post("/api/tickets", json=wrong).status_code, 404)


if __name__ == "__main__":
    unittest.main()
