"""Archive invariants: immutable bytes, causal times, no outcomes, atomic retries."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from api import forecast_store as store


def payload(run_id="object_2026-06-28_abc"):
    return {"run_id": run_id, "entity_mode": "object", "mode": "historical_replay",
            "feature_date": "2026-06-28", "feature_cutoff": "2026-06-28T23:59:59+03:00",
            "issue_time": "2026-06-29T00:00:00+03:00", "forecast_start": "2026-06-30T00:00:00+03:00",
            "forecast_end": "2026-07-01T00:00:00+03:00", "minimum_lead_hours": 24,
            "model_sha256": "a" * 64, "input_sha256": "b" * 64, "threshold": .4,
            "cards": [{"id": f"{run_id}:object:1", "object_id": "1", "entity_mode": "object",
                       "score": .6, "warning": True, "object_name": "Объект 1"}]}


class ForecastStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "archive.sqlite3"

    def tearDown(self):
        self.tmp.cleanup()

    def test_canonical_retry_preserves_original_creation_time_and_payload(self):
        value = payload()
        meta = store.publish_run(self.db, value)
        reordered = dict(reversed(list(value.items())))
        self.assertEqual(meta, store.publish_run(self.db, reordered))
        self.assertEqual(store.load_run(self.db, value["run_id"], "object"), value)
        self.assertEqual(store.get_run_metadata(self.db, value["run_id"]), meta)
        self.assertEqual(meta["cards_count"], 1)
        self.assertEqual(len(meta["content_sha256"]), 64)
        self.assertNotEqual(meta["created_at"], meta["issue_time"])

    def test_changed_release_and_cross_release_card_collision_roll_back(self):
        value = payload("run_a")
        store.publish_run(self.db, value)
        changed = deepcopy(value)
        changed["cards"][0]["score"] = .7
        with self.assertRaises(store.ForecastConflictError):
            store.publish_run(self.db, changed)
        # Both ids contain "run_a"; a second run cannot reuse the existing card.
        conflict = deepcopy(value)
        conflict["run_id"] = "run"
        with self.assertRaises(store.ForecastConflictError):
            store.publish_run(self.db, conflict)
        self.assertEqual(len(store.list_runs(self.db)), 1)
        self.assertEqual(store.load_run(self.db, "run_a")["cards"][0]["score"], .6)

    def test_database_rejects_update_and_delete(self):
        value = payload()
        store.publish_run(self.db, value)
        with sqlite3.connect(self.db) as db:
            for statement in ["UPDATE forecast_runs SET content_sha256='x'", "DELETE FROM forecast_runs",
                              "UPDATE forecast_card_ids SET card_id='x'", "DELETE FROM forecast_card_ids"]:
                with self.subTest(statement=statement), self.assertRaises(sqlite3.IntegrityError):
                    db.execute(statement)

    def test_integrity_checks_payload_hash_metadata_and_future_outcome(self):
        value = payload()
        store.publish_run(self.db, value)
        with sqlite3.connect(self.db) as db:
            db.execute("DROP TRIGGER forecast_runs_immutable_update")
            db.execute("UPDATE forecast_runs SET payload_json='{}'")
        with self.assertRaises(store.ForecastIntegrityError):
            store.load_run(self.db, value["run_id"])
        with self.assertRaises(store.ForecastIntegrityError):
            store.list_runs(self.db)

    def test_nested_outcomes_and_nonfinite_values_never_enter_archive(self):
        for key, value in [("actual_target_alarm", True), ("actual_hidden", 0), ("outcome", False),
                           ("nested", {"list": [{"actual_next_day_alarm": True}]}), ("value", float("nan")),
                           ("value", float("inf"))]:
            candidate = payload()
            candidate["cards"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(store.ForecastValidationError):
                store.publish_run(self.db, candidate)
        self.assertFalse(self.db.exists())

    def test_rejects_invalid_temporal_contract(self):
        variations = [{"feature_cutoff": "2026-06-28T23:59:59"},
                      {"issue_time": "2026-06-29T00:00:00"},
                      {"feature_cutoff": "2026-06-29T00:00:00+03:00"},
                      {"issue_time": "2026-06-28T23:59:59+03:00"},
                      {"forecast_start": "2026-06-29T23:59:59+03:00"},
                      {"minimum_lead_hours": 48},
                      {"forecast_end": "2026-06-30T00:00:00+03:00"},
                      {"window_hours": 48}, {"feature_date": "20260628"}]
        for changes in variations:
            with self.subTest(changes=changes), self.assertRaises(store.ForecastValidationError):
                store.publish_run(self.db, {**payload(), **changes})

    def test_scores_warning_counts_ids_and_mode_are_consistent(self):
        values = []
        for field, value in [("score", 1.1), ("score", True), ("score", "0.6"),
                             ("warning", False), ("warning", 1), ("id", "missing_run_id"), ("entity_mode", "channel")]:
            candidate = payload()
            candidate["cards"][0][field] = value
            values.append(candidate)
        candidate = payload()
        candidate["cards"].append(deepcopy(candidate["cards"][0]))
        values.append(candidate)
        values += [{**payload(), field: value} for field, value in [("mode", "live"), ("entity_mode", "channel"),
                   ("threshold", -.1), ("model_sha256", "abc"), ("warnings_count", 0), ("shown_cards", 2)]]
        for i, candidate in enumerate(values):
            with self.subTest(i=i), self.assertRaises(store.ForecastValidationError):
                store.publish_run(self.db, candidate)

    def test_empty_legacy_database_and_mode_mismatch(self):
        self.assertEqual(store.list_runs(self.db), [])
        self.assertFalse(self.db.exists())
        with sqlite3.connect(self.db) as db:
            db.execute("CREATE TABLE tickets (id TEXT)")
        self.assertEqual(store.list_runs(self.db), [])
        store.publish_run(self.db, payload())
        with self.assertRaises(store.ForecastNotFoundError):
            store.load_run(self.db, payload()["run_id"], "channel")
        with self.assertRaises(store.ForecastNotFoundError):
            store.load_run(self.db, "unknown")
        self.assertEqual(store.list_runs(self.db, "channel"), [])

    def test_concurrent_retries_publish_exactly_one_release(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: store.publish_run(self.db, payload()), range(8)))
        self.assertTrue(all(meta == results[0] for meta in results))
        self.assertEqual(len(store.list_runs(self.db)), 1)


if __name__ == "__main__":
    unittest.main()
