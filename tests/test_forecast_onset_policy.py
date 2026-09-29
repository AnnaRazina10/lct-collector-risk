"""Onset top-k policy and byte-compatible legacy release regression checks."""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from api import forecast_store as store
from tests.test_forecast_store import payload


def onset_payload(run_id="onset_2026-06-28_abc"):
    result = payload(run_id)
    result.update(target_kind="registered_episode_start_g1",
                  target_definition="Начало зарегистрированной серии на D+2 после суток D+1 без тревожных записей.",
                  warning_policy={"kind": "daily_top_k", "k": 10, "tie_break": "score_desc_object_id_asc"},
                  score_kind="uncalibrated_onset_score", threshold=.5, warnings_count=10)
    identities = [str(n) for n in range(1, 13)]
    rank = {oid: position for position, oid in enumerate(sorted(identities), start=1)}
    result["cards"] = [{"id": f"{run_id}:object:{oid}", "object_id": oid, "entity_mode": "object",
                        "object_name": f"Объект {oid}", "score": .5, "rank": rank[oid],
                        "warning": rank[oid] <= 10} for oid in identities]
    return result


class OnsetPolicyStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "archive.sqlite3"

    def tearDown(self):
        self.tmp.cleanup()

    def test_boundary_ties_select_exactly_ten_by_string_id_not_threshold(self):
        candidate = onset_payload()
        candidate["cards"].reverse()  # Payload array order does not determine rank.
        meta = store.publish_run(self.db, candidate)
        loaded = store.load_run(self.db, candidate["run_id"])
        cards = {card["object_id"]: card for card in loaded["cards"]}
        self.assertEqual(cards["10"]["rank"], 2)
        self.assertTrue(cards["12"]["warning"])
        self.assertFalse(cards["8"]["warning"])
        self.assertFalse(cards["9"]["warning"])
        self.assertEqual(sum(c["score"] >= candidate["threshold"] for c in cards.values()), 12)
        self.assertEqual(sum(c["warning"] for c in cards.values()), 10)
        self.assertEqual(meta, store.publish_run(self.db, candidate))
        for key in ("target_kind", "target_definition", "warning_policy", "score_kind"):
            self.assertEqual(meta[key], candidate[key])

    def test_optional_score_kind_is_not_invented(self):
        candidate = onset_payload()
        del candidate["score_kind"]
        meta = store.publish_run(self.db, candidate)
        self.assertNotIn("score_kind", meta)
        self.assertNotIn("score_kind", store.load_run(self.db, candidate["run_id"]))

    def test_rejects_unsupported_or_ambiguous_policy(self):
        policies = [None, {}, {"kind": "daily_top_k", "k": 10},
                    {"kind": "threshold", "k": 10, "tie_break": "score_desc_object_id_asc"},
                    {"kind": "daily_top_k", "k": 10, "tie_break": "numeric_object_id"},
                    {"kind": "daily_top_k", "k": 10, "tie_break": "score_desc_object_id_asc", "other": 1}]
        policies += [{"kind": "daily_top_k", "k": k, "tie_break": "score_desc_object_id_asc"} for k in [5, 20, 0, -1, True, 10.0, "10"]]
        for policy in policies:
            with self.subTest(policy=policy), self.assertRaises(store.ForecastValidationError):
                store.validate_payload({**onset_payload(), "warning_policy": policy})
        no_policy = onset_payload(); del no_policy["warning_policy"]
        with self.assertRaises(store.ForecastValidationError):
            store.validate_payload(no_policy)

    def test_target_definition_rank_and_object_identity_are_required(self):
        for field, value in [("target_kind", "any_alarm"), ("target_definition", " "),
                             ("target_definition", None), ("score_kind", "")]:
            with self.subTest(field=field, value=value), self.assertRaises(store.ForecastValidationError):
                store.validate_payload({**onset_payload(), field: value})
        for field, values in [("rank", [None, 0, 13, True, 1.0]),
                              ("object_id", [None, "", "  ", True, {}, 1.5])]:
            for value in values:
                candidate = onset_payload(); candidate["cards"][0][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(store.ForecastValidationError):
                    store.validate_payload(candidate)
        candidate = onset_payload(); candidate["cards"][1]["object_id"] = 1
        with self.assertRaisesRegex(store.ForecastValidationError, "Duplicate object_id"):
            store.validate_payload(candidate)
        candidate = onset_payload(); del candidate["cards"][0]["object_id"]
        with self.assertRaises(store.ForecastValidationError):
            store.validate_payload(candidate)

    def test_exact_ranks_budget_and_boundary_score_are_enforced(self):
        candidates = []
        candidate = onset_payload(); candidate["threshold"] = .49; candidates.append(candidate)
        candidate = onset_payload(); candidate["cards"] = candidate["cards"][:9]; candidates.append(candidate)
        candidate = onset_payload(); candidate["cards"][0]["warning"] = False; candidates.append(candidate)
        candidate = onset_payload(); candidate["cards"][7]["warning"] = True; candidates.append(candidate)
        candidate = onset_payload(); candidate["cards"][0]["rank"] = 2; candidates.append(candidate)
        candidate = onset_payload(); candidate["warnings_count"] = 11; candidates.append(candidate)
        # A numeric-id rank order is wrong even when it gives exactly ten warnings.
        candidate = onset_payload()
        for rank, card in enumerate(candidate["cards"], start=1):
            card["rank"] = rank; card["warning"] = rank <= 10
        candidates.append(candidate)
        for i, candidate in enumerate(candidates):
            with self.subTest(i=i), self.assertRaises(store.ForecastValidationError):
                store.validate_payload(candidate)

    def test_future_onset_labels_are_rejected_at_every_nested_depth(self):
        names = ["onset_target", "alarm_target", "joint_target", "quiet_intermediate",
                 "records_intermediate", "records_target", "ONSET_TARGET"]
        for factory in (payload, onset_payload):
            for name in names:
                candidate = factory(); candidate["cards"][0]["explanation"] = [{"nested": {name: 0}}]
                with self.subTest(name=name, mode=factory.__name__), self.assertRaises(store.ForecastValidationError):
                    store.publish_run(self.db, candidate)
        self.assertFalse(self.db.exists())

    def test_legacy_metadata_and_canonical_bytes_remain_unchanged(self):
        legacy = payload("old_run")
        legacy["score_kind"] = "uncalibrated_model_score"  # Already present in real old payloads.
        raw = json.dumps(legacy, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        digest = hashlib.sha256(raw.encode()).hexdigest()
        fields = ("run_id", "entity_mode", "mode", "feature_date", "feature_cutoff", "issue_time",
                  "forecast_start", "forecast_end", "minimum_lead_hours", "model_sha256", "input_sha256", "threshold")
        created = "2026-09-29T00:00:00+00:00"
        old_meta = {**{key: legacy[key] for key in fields}, "cards_count": 1, "content_sha256": digest, "created_at": created}
        encoded_meta = json.dumps(old_meta, sort_keys=True, ensure_ascii=False)
        with sqlite3.connect(self.db) as db:
            db.execute("CREATE TABLE forecast_runs (run_id TEXT PRIMARY KEY, entity_mode TEXT, mode TEXT, issue_epoch REAL, metadata_json TEXT, payload_json TEXT, content_sha256 TEXT, created_at TEXT)")
            db.execute("INSERT INTO forecast_runs VALUES (?,?,?,?,?,?,?,?)", (legacy["run_id"], "object", "historical_replay",
                       datetime.fromisoformat(legacy["issue_time"]).timestamp(), encoded_meta, raw, digest, created))
        self.assertEqual(store.load_run(self.db, "old_run"), legacy)
        self.assertEqual(store.list_runs(self.db), [old_meta])
        self.assertEqual(store.publish_run(self.db, legacy), old_meta)
        self.assertNotIn("score_kind", store.get_run_metadata(self.db, "old_run"))
        with sqlite3.connect(self.db) as db:
            row = db.execute("SELECT metadata_json,payload_json,content_sha256 FROM forecast_runs").fetchone()
        self.assertEqual(row, (encoded_meta, raw, digest))
        wrong = deepcopy(legacy); wrong["cards"][0]["warning"] = False
        with self.assertRaises(store.ForecastValidationError):
            store.validate_payload(wrong)

    def test_read_revalidates_future_fields_even_after_checksum_is_recomputed(self):
        candidate = onset_payload(); meta = store.publish_run(self.db, candidate)
        candidate["cards"][0]["nested"] = {"records_target": 1}
        raw = json.dumps(candidate, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(raw.encode()).hexdigest(); meta["content_sha256"] = digest
        with sqlite3.connect(self.db) as db:
            db.execute("DROP TRIGGER forecast_runs_immutable_update")
            db.execute("UPDATE forecast_runs SET payload_json=?,content_sha256=?,metadata_json=?",
                       (raw, digest, json.dumps(meta, ensure_ascii=False, sort_keys=True)))
        with self.assertRaises(store.ForecastIntegrityError):
            store.load_run(self.db, candidate["run_id"])


if __name__ == "__main__":
    unittest.main()
