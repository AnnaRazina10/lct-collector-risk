"""Past-only recommendation boundary, not model-quality or repair validation."""
import copy
from datetime import date, datetime, time, timedelta, timezone
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/serving"))
import local_recommendations as rec


def context(goal="any_alarm", day="2026-06-28", object_id="4767"):
    start = datetime.combine(date.fromisoformat(day), time(), tzinfo=timezone(timedelta(hours=3)))
    run_id = ("onset_" if goal == "registered_episode_start_g1" else "object_") + day + "_test"
    return {"run_id": run_id, "card_id": f"{run_id}_{object_id}", "object_id": object_id,
            "goal": goal, "mode": "historical_replay", "feature_date": day,
            "feature_cutoff": (start+timedelta(days=1, seconds=-1)).isoformat(),
            "issue_time": (start+timedelta(days=1)).isoformat(),
            "forecast_start": (start+timedelta(days=2)).isoformat(),
            "forecast_end": (start+timedelta(days=3)).isoformat()}


def facts(**changes):
    value = {"observed_channels": 3, "catalog_channels": 3, "events_today": 109598,
             "alarm_channels": 0, "current_alarm": False, "warning": True,
             "alarm_days_7d": 1, "observed_days_7d": 7}
    return dict(value, **changes)


class LocalRecommendationTests(unittest.TestCase):
    def check_rules(self, values, expected, goal="any_alarm"):
        result = rec.recommend(values, context(goal))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["rule_ids"], expected)
        self.assertEqual([step["rule_id"] for step in result["steps"]], expected)
        self.assertLessEqual(len(result["steps"]), 3)
        self.assertTrue(result["requires_dispatcher_decision"])
        self.assertFalse(result["external_send"])
        return result

    def test_no_record_rule_is_exclusive_without_claiming_health(self):
        result = self.check_rules(facts(observed_channels=0, events_today=0,
            alarm_days_7d=4, observed_days_7d=6), ["NO_RECORDS_TODAY"])
        self.assertIn("состояние оборудования не установлено", result["steps"][0]["text"])

    def test_partial_catalog_is_not_a_failed_sensor_claim(self):
        result = self.check_rules(facts(catalog_channels=5), ["PARTIAL_CATALOG_RECORDS"])
        self.assertIn("3 из 5", result["steps"][0]["text"])
        self.assertIn("не доказывает неисправность", result["steps"][0]["text"])

    def test_current_alarm_preserves_selected_onset_object(self):
        result = self.check_rules(facts(current_alarm=True, alarm_channels=1, rank=4),
            ["ALARM_RECORDED_TODAY"], "registered_episode_start_g1")
        evidence = {item["field"]: item["value"] for item in result["evidence"]}
        self.assertTrue(evidence["warning"])
        self.assertEqual(evidence["rank"], 4)

    def test_recurring_alarm_days_are_not_event_counts(self):
        result = self.check_rules(facts(alarm_days_7d=4), ["MULTIPLE_ALARM_DAYS"])
        self.assertIn("в 4 днях", result["steps"][0]["text"])

    def test_rule_order_and_three_independent_steps(self):
        self.check_rules(facts(catalog_channels=5, current_alarm=True, alarm_channels=2,
            alarm_days_7d=3), ["PARTIAL_CATALOG_RECORDS", "ALARM_RECORDED_TODAY", "MULTIPLE_ALARM_DAYS"])

    def test_selected_fallback(self):
        self.check_rules(facts(), ["SELECTED_FOR_REVIEW"])

    def test_outside_queue_is_not_a_health_statement(self):
        result = self.check_rules(facts(warning=False, rank=11), ["OUTSIDE_WARNING_POLICY"],
            "registered_episode_start_g1")
        self.assertIn("не подтверждает исправность", result["steps"][0]["text"])

    def test_any_alarm_does_not_require_rank(self):
        result = self.check_rules(facts(), ["SELECTED_FOR_REVIEW"])
        self.assertNotIn("rank", [item["field"] for item in result["evidence"]])

    def test_large_event_counts_are_not_artificially_capped(self):
        self.check_rules(facts(events_today=10**30), ["SELECTED_FOR_REVIEW"])

    def test_every_missing_known_fact_returns_insufficient_data(self):
        for field in rec.FACT_FIELDS:
            value = facts(); del value[field]
            with self.subTest(field=field):
                result = rec.recommend(value, context())
                self.assertEqual(result["status"], "insufficient_data")
                self.assertEqual(result["steps"], [])
                self.assertEqual(result["evidence"], [])
                self.assertIn({"field": field, "code": "missing"}, result["validation_errors"])

    def test_invalid_known_fact_types_never_become_evidence(self):
        for field, invalid in [("current_alarm", 1), ("warning", "yes"),
            ("events_today", True), ("events_today", 1.0), ("events_today", float("nan")),
            ("events_today", float("inf")), ("events_today", "<script>"),
            ("events_today", []), ("observed_channels", None), ("rank", 0), ("rank", False)]:
            with self.subTest(field=field, invalid=invalid):
                result = rec.recommend(facts(**{field: invalid}), context())
                self.assertEqual(result["status"], "insufficient_data")
                self.assertEqual(result["steps"], [])
                self.assertEqual(result["evidence"], [])
                json.dumps(result, allow_nan=False)

    def test_inconsistent_counters_fail_closed(self):
        cases = [
            ({"events_today": -1}, "expected_nonnegative_integer"),
            ({"alarm_days_7d": 8}, "outside_seven_day_window"),
            ({"catalog_channels": 2}, "exceeds_catalog_channels"),
            ({"alarm_channels": 4}, "exceeds_observed_channels"),
            ({"events_today": 2}, "fewer_events_than_observed_channels"),
            ({"observed_channels": 0}, "inconsistent_with_observed_channels"),
            ({"current_alarm": True}, "inconsistent_with_alarm_channels"),
            ({"alarm_channels": 1}, "inconsistent_with_alarm_channels"),
            ({"alarm_days_7d": 3, "observed_days_7d": 2}, "exceeds_observed_days"),
            ({"current_alarm": True, "alarm_channels": 1, "alarm_days_7d": 0}, "omits_current_alarm_day"),
            ({"alarm_days_7d": 0, "observed_days_7d": 0}, "omits_current_observed_day"),
            ({"observed_channels": 0, "events_today": 0}, "includes_unobserved_current_day"),
            ({"alarm_days_7d": 7}, "includes_unalarmed_current_day"),
        ]
        for changes, expected in cases:
            with self.subTest(changes=changes):
                result = rec.recommend(facts(**changes), context())
                self.assertEqual(result["status"], "insufficient_data")
                self.assertEqual(result["rule_ids"], [])
                self.assertIn(expected, [error["code"] for error in result["validation_errors"]])

    def test_future_fields_and_unapproved_inputs_are_rejected(self):
        for field in ["actual_alarm", "actual_onset", "outcome", "records_intermediate",
                      "records_target", "onset_target", "score", "explanation", "note", "no_power"]:
            with self.subTest(field=field), self.assertRaises(rec.RecommendationValidationError):
                rec.recommend(facts(**{field: 1}), context())
        with self.assertRaises(rec.RecommendationValidationError):
            rec.recommend(facts(), dict(context(), generated_at="2026-09-29T12:00:00Z"))

    def test_invalid_input_containers_and_nonstring_keys_are_rejected(self):
        for value, ctx in [(None, context()), ([], context()), (facts(), None),
                           ({1: "unknown"}, context())]:
            with self.subTest(value=value), self.assertRaises(rec.RecommendationValidationError):
                rec.recommend(value, ctx)

    def test_incomplete_or_unsupported_context_is_rejected(self):
        for key in rec.CONTEXT_FIELDS:
            ctx = context(); del ctx[key]
            with self.subTest(key=key), self.assertRaises(rec.RecommendationValidationError):
                rec.recommend(facts(), ctx)
        for key, value in [("mode", "live"), ("goal", "fire"), ("object_id", ""),
                           ("feature_date", "not-a-date"), ("feature_date", "20260628")]:
            with self.subTest(key=key), self.assertRaises(rec.RecommendationValidationError):
                rec.recommend(facts(), dict(context(), **{key: value}))

    def test_card_cannot_be_silently_bound_to_another_release_or_goal(self):
        for changes in [{"card_id": "other"}, {"object_id": "5582"},
                        {"run_id": "object_other"}, {"goal": "registered_episode_start_g1"}]:
            with self.subTest(changes=changes), self.assertRaises(rec.RecommendationValidationError):
                rec.recommend(facts(), dict(context(), **changes))

    def test_temporal_boundary_rejects_future_cutoff_short_lead_and_wrong_window(self):
        cases = [("feature_cutoff", "2026-06-29T00:00:00+03:00"),
                 ("feature_cutoff", "2026-06-27T23:59:59+03:00"),
                 ("issue_time", "2026-06-29T00:01:00+03:00"),
                 ("forecast_start", "2026-06-29T23:59:59+03:00"),
                 ("forecast_start", "2026-07-01T00:00:00+03:00"),
                 ("forecast_end", "2026-07-01T00:00:01+03:00"),
                 ("issue_time", "2026-06-29T00:00:00"),
                 ("issue_time", "yesterday"),
                 ("issue_time", "9999-12-31T23:00:00Z")]
        for key, value in cases:
            with self.subTest(key=key, value=value), self.assertRaises(rec.RecommendationValidationError):
                rec.recommend(facts(), dict(context(), **{key: value}))

    def test_equivalent_utc_context_produces_same_canonical_result(self):
        ctx = context()
        for field in ["feature_cutoff", "issue_time", "forecast_start", "forecast_end"]:
            ctx[field] = datetime.fromisoformat(ctx[field]).astimezone(timezone.utc).isoformat()
        self.assertEqual(rec.recommend(facts(), context()), rec.recommend(facts(), ctx))

    def test_evidence_separates_observation_cutoff_from_policy_availability(self):
        value = facts(current_alarm=True, alarm_channels=1, alarm_days_7d=3, rank=10)
        result = rec.recommend(value, context("registered_episode_start_g1"))
        evidence = {item["field"]: item["value"] for item in result["evidence"]}
        self.assertEqual(evidence, value)
        for item in result["evidence"]:
            if item["field"] in {"warning", "rank"}:
                self.assertEqual(item["source"], "forecast_policy")
                self.assertEqual(item["available_at"], result["issue_time"])
                self.assertNotIn("window_end", item)
            else:
                self.assertEqual(item["source"], "observation")
                self.assertEqual(item["window_end"], result["fact_cutoff"])
                self.assertNotIn("available_at", item)
        for step in result["steps"]:
            self.assertTrue(set(step["evidence_fields"]) <= evidence.keys())

    def test_identity_includes_release_goal_and_unused_accepted_facts(self):
        base = rec.recommend(facts(), context())
        variants = [rec.recommend(facts(), context(day="2026-06-27")),
                    rec.recommend(facts(), context("registered_episode_start_g1")),
                    rec.recommend(facts(events_today=109599), context())]
        for value in variants:
            self.assertNotEqual(base["recommendation_id"], value["recommendation_id"])
        self.assertRegex(base["recommendation_id"], r"^[0-9a-f]{64}$")
        self.assertEqual(base["ruleset_sha256"], rec.RULESET_SHA256)

    def test_deterministic_without_mutating_input_or_reusing_output_containers(self):
        value, ctx = facts(), context()
        before = copy.deepcopy((value, ctx))
        first, second = rec.recommend(value, ctx), rec.recommend(dict(reversed(list(value.items()))), ctx)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        self.assertEqual((value, ctx), before)
        first["evidence"][0]["value"] = "mutated"
        first["limitations"].clear()
        self.assertEqual(second, rec.recommend(value, ctx))
        self.assertNotIn("generated_at", second)

    def test_pure_module_preserves_a_78_card_fixed_top10_queue(self):
        cards = [{"id": context("registered_episode_start_g1", object_id=str(i))["card_id"],
                  "object_id": str(i), "score": 0.5,
                  **facts(warning=i <= 10, rank=i, current_alarm=i == 1,
                          alarm_channels=int(i == 1))} for i in range(1, 79)]
        original = copy.deepcopy(cards)
        outputs = [rec.recommend({key: card[key] for key in rec.FACT_FIELDS | rec.OPTIONAL_FACT_FIELDS},
                   context("registered_episode_start_g1", object_id=card["object_id"])) for card in cards]
        self.assertEqual(cards, original)
        self.assertEqual(sum(card["warning"] for card in cards), 10)
        self.assertEqual(len({result["recommendation_id"] for result in outputs}), 78)
        self.assertTrue(all(result["status"] == "ok" for result in outputs))


if __name__ == "__main__":
    unittest.main()
