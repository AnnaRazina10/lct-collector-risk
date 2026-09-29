"""Release choice persists through API read, draft and dispatcher decisions."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from api import forecast_store, main
from tests.test_forecast_store import payload


class ForecastApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        folder = Path(self.tmp.name)
        self.db = folder / "app.sqlite3"
        legacy = folder / "legacy.json"
        legacy.write_text(json.dumps({"cards": [{"id": "old", "score": .3, "actual_target_alarm": True}]}))
        self.patch = patch.multiple(main, DB=self.db, DATA=legacy, OBJECT_DATA=legacy)
        self.patch.start()
        self.client = TestClient(main.app)
        self.first = payload("run_1")
        self.second = payload("run_2")
        self.second.update(feature_date="2026-06-29", feature_cutoff="2026-06-29T23:59:59+03:00",
                           issue_time="2026-06-30T00:00:00+03:00", forecast_start="2026-07-01T00:00:00+03:00",
                           forecast_end="2026-07-02T00:00:00+03:00")
        forecast_store.publish_run(self.db, self.first)
        forecast_store.publish_run(self.db, self.second)

    def tearDown(self):
        self.client.close()
        self.patch.stop()
        self.tmp.cleanup()

    def test_select_release_and_keep_legacy_routes(self):
        listed = self.client.get("/api/forecast-runs?mode=object").json()["runs"]
        self.assertEqual([r["run_id"] for r in listed], ["run_2", "run_1"])
        release = self.client.get("/api/risks?mode=object&run_id=run_1").json()
        self.assertEqual(release["issue_time"], self.first["issue_time"])
        self.assertEqual(release["archive_metadata"]["content_sha256"], listed[1]["content_sha256"])
        self.assertNotIn("actual_target_alarm", json.dumps(release))
        cid = self.first["cards"][0]["id"]
        detail = self.client.get(f"/api/risks/{cid}?mode=object&run_id=run_1").json()
        self.assertEqual(detail["forecast"]["run_id"], "run_1")
        self.assertEqual(self.client.get(f"/api/risks/{cid}/outcome?mode=object&run_id=run_1").status_code, 404)
        self.assertEqual(self.client.get("/api/risks?mode=object").json()["cards"][0]["id"], "old")
        self.assertTrue(self.client.get("/api/risks/old/outcome?mode=object").json()["actual_alarm"])

    def test_unknown_run_mode_mismatch_and_cross_release_card_are_404(self):
        cid = self.first["cards"][0]["id"]
        for path in ["/api/risks?mode=object&run_id=absent", "/api/risks?mode=channel&run_id=run_1",
                     f"/api/risks/{cid}?mode=object&run_id=run_2", f"/api/risks/{cid}?mode=object"]:
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 404)
        self.assertEqual(self.client.get("/api/forecast-runs?mode=channel").json()["runs"], [])
        self.assertEqual(self.client.get("/api/forecast-runs?limit=501").status_code, 422)

    def test_tickets_and_feedback_remain_bound_to_selected_release(self):
        for release in (self.first, self.second):
            body = {"risk_id": release["cards"][0]["id"], "entity_mode": "object", "run_id": release["run_id"]}
            draft = self.client.post("/api/tickets", json=body)
            self.assertEqual(draft.status_code, 201)
            self.assertEqual(draft.json()["risk_snapshot"]["forecast"]["run_id"], release["run_id"])
            decision = self.client.post("/api/feedback", json={**body, "operator": "Демо", "decision": "monitor", "reason": "other"})
            self.assertEqual(decision.status_code, 201)
            snap = decision.json()["risk_snapshot"]
            self.assertEqual(snap["forecast"]["issue_time"], release["issue_time"])
            self.assertEqual(snap["forecast"]["input_sha256"], release["input_sha256"])
        entries = self.client.get("/api/journal?mode=object").json()["entries"]
        self.assertEqual(len(entries), 4)
        self.assertEqual({e["risk_snapshot"]["forecast"]["run_id"] for e in entries}, {"run_1", "run_2"})
        self.assertNotIn("actual_", json.dumps(entries))
        wrong = {"risk_id": self.first["cards"][0]["id"], "entity_mode": "object", "run_id": "run_2"}
        self.assertEqual(self.client.post("/api/tickets", json=wrong).status_code, 404)

    def test_corruption_is_fail_closed_and_never_falls_back_to_legacy(self):
        with sqlite3.connect(self.db) as db:
            db.execute("DROP TRIGGER forecast_runs_immutable_update")
            db.execute("UPDATE forecast_runs SET payload_json='{}' WHERE run_id='run_1'")
        self.assertEqual(self.client.get("/api/risks?mode=object&run_id=run_1").status_code, 503)
        self.assertEqual(self.client.get("/api/forecast-runs").status_code, 503)


if __name__ == "__main__":
    unittest.main()
