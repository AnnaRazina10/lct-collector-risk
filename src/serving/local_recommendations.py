"""Deterministic review suggestions from a small, past-only card projection.

No I/O, model execution, clock, persistence, or equipment actions. The caller must
resolve the card from a validated server-side release and verify input provenance.
Dates alone cannot prove that the underlying observations were available then.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json

SCHEMA_VERSION = "1.0"
RULESET_VERSION = "1.0.0"
MSK = timezone(timedelta(hours=3))
FACT_FIELDS = frozenset({"observed_channels", "catalog_channels", "events_today",
    "alarm_channels", "current_alarm", "warning", "alarm_days_7d", "observed_days_7d"})
OPTIONAL_FACT_FIELDS = frozenset({"rank"})
CONTEXT_FIELDS = frozenset({"run_id", "card_id", "object_id", "goal", "mode",
    "feature_date", "feature_cutoff", "issue_time", "forecast_start", "forecast_end"})
GOAL_PREFIXES = {"any_alarm": "object_", "registered_episode_start_g1": "onset_"}

# This specification is part of the audit identity. Bump the version when the
# conditions, validation semantics, order, or texts change.
_RULES = (
    ("NO_RECORDS_TODAY", "observed_channels == 0; exclusive",
     ("observed_channels", "events_today"),
     "За день признаков записи не зарегистрированы. Проверьте доступность источника данных "
     "и журнал поступления. По отсутствию записей состояние оборудования не установлено."),
    ("PARTIAL_CATALOG_RECORDS", "0 < observed_channels < catalog_channels",
     ("observed_channels", "catalog_channels"),
     "Записи есть по {observed_channels} из {catalog_channels} справочных каналов. "
     "Уточните ожидаемый режим регистрации и полноту поступления; отсутствие записей "
     "по каналу не доказывает неисправность."),
    ("ALARM_RECORDED_TODAY", "current_alarm is True",
     ("current_alarm", "alarm_channels"),
     "В день признаков уже были тревожные записи. Сопоставьте их с журналом диспетчера "
     "и ранее принятыми решениями; необходимость осмотра определяет диспетчер."),
    ("MULTIPLE_ALARM_DAYS", "alarm_days_7d >= 2",
     ("alarm_days_7d", "observed_days_7d"),
     "За последние семь календарных дней тревожные записи отмечены в {alarm_days_7d} днях. "
     "Проверьте их повторяемость и предыдущую отработку по журналу."),
    ("SELECTED_FOR_REVIEW", "no preceding rule; warning is True", ("warning",),
     "Объект включён в очередь проверки. Сопоставьте доступные наблюдения и журнал "
     "диспетчера; модельная оценка сама по себе не подтверждает поломку."),
    ("OUTSIDE_WARNING_POLICY", "no preceding rule; warning is False", ("warning",),
     "Объект не выбран действующей политикой предупреждений. Отсутствие предупреждения "
     "не подтверждает исправность; дополнительных оснований для рекомендации "
     "в переданных фактах нет."),
)
_LIMITATIONS = (
    "Предложены только шаги проверки доступной информации; физическая причина не установлена.",
    "Историческое воспроизведение, не действующий поток. Правила применены к архивным фактам.",
    "Отсутствие записей или предупреждения не подтверждает исправность оборудования.",
    "Окончательное решение принимает диспетчер; отправка заявок и управление оборудованием отсутствуют.",
)


class RecommendationValidationError(ValueError):
    """An untrusted field, identity, or temporal context crossed the boundary."""


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


RULESET_SHA256 = _sha({"schema_version": SCHEMA_VERSION, "ruleset_version": RULESET_VERSION,
    "rules": _RULES, "max_steps": 3, "limitations": _LIMITATIONS,
    "fact_fields": sorted(FACT_FIELDS), "optional_fact_fields": sorted(OPTIONAL_FACT_FIELDS),
    "context_fields": sorted(CONTEXT_FIELDS), "goal_prefixes": GOAL_PREFIXES,
    "validation": "strict_json_types;consistent_nonnegative_counters;daily_MSK_D_plus_2;v1",
    "evidence_timing": {"observation": "window_end=feature_cutoff",
                        "forecast_policy": "available_at=issue_time"}})


def _project(value: Mapping[str, object], allowed: frozenset[str], name: str) -> dict:
    if not isinstance(value, Mapping):
        raise RecommendationValidationError(f"{name} must be a mapping")
    if any(type(key) is not str for key in value) or set(value) - allowed:
        # Do not echo arbitrary client values or unknown field content.
        raise RecommendationValidationError(f"Unknown {name} fields are forbidden")
    return dict(value)


def _timestamp(value: object, field: str) -> datetime:
    if type(value) is not str:
        raise RecommendationValidationError(f"{field} must be an aware ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RecommendationValidationError(f"Invalid {field}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RecommendationValidationError(f"{field} requires a timezone")
    try:
        return parsed.astimezone(MSK)
    except OverflowError as exc:
        raise RecommendationValidationError(f"Invalid {field}") from exc


def _context(value: Mapping[str, object]) -> dict:
    context = _project(value, CONTEXT_FIELDS, "context")
    if set(context) != CONTEXT_FIELDS:
        raise RecommendationValidationError("All context fields are required")
    if any(type(context[key]) is not str or not context[key].strip() for key in CONTEXT_FIELDS):
        raise RecommendationValidationError("Context values must be nonempty strings")
    if context["mode"] != "historical_replay" or context["goal"] not in GOAL_PREFIXES:
        raise RecommendationValidationError("Unsupported mode or goal")
    if not context["run_id"].startswith(GOAL_PREFIXES[context["goal"]]):
        raise RecommendationValidationError("Run prefix differs from the explicit goal")
    if context["card_id"] != f'{context["run_id"]}_{context["object_id"]}':
        raise RecommendationValidationError("Card identity differs from run and object")
    try:
        feature_date = date.fromisoformat(context["feature_date"])
        if feature_date.isoformat() != context["feature_date"]:
            raise ValueError("noncanonical feature date")
        day = datetime.combine(feature_date, time(), tzinfo=MSK)
        issue = day + timedelta(days=1)
        expected = {"feature_cutoff": issue-timedelta(seconds=1), "issue_time": issue,
                    "forecast_start": day+timedelta(days=2), "forecast_end": day+timedelta(days=3)}
    except (ValueError, OverflowError) as exc:
        raise RecommendationValidationError("Invalid feature_date") from exc
    # Version 1 accepts the existing daily D+2 contract, not arbitrary windows.
    for field, expected_time in expected.items():
        parsed = _timestamp(context[field], field)
        if parsed != expected_time:
            raise RecommendationValidationError(f"{field} violates the daily 24-hour lead contract")
        context[field] = parsed.isoformat()
    return context


def _fact_errors(facts: dict) -> list[dict[str, str]]:
    errors = []

    def error(field: str, code: str) -> None:
        errors.append({"field": field, "code": code})

    for field in sorted(FACT_FIELDS):
        if field not in facts:
            error(field, "missing")
        elif field in {"current_alarm", "warning"}:
            if type(facts[field]) is not bool:
                error(field, "expected_boolean")
        elif type(facts[field]) is not int or facts[field] < 0:
            error(field, "expected_nonnegative_integer")
        elif field.endswith("_7d") and facts[field] > 7:
            error(field, "outside_seven_day_window")
    if "rank" in facts and (type(facts["rank"]) is not int or facts["rank"] < 1):
        error("rank", "expected_positive_integer")
    if errors:
        return errors

    if facts["observed_channels"] > facts["catalog_channels"]:
        error("observed_channels", "exceeds_catalog_channels")
    if facts["alarm_channels"] > facts["observed_channels"]:
        error("alarm_channels", "exceeds_observed_channels")
    if facts["events_today"] < facts["observed_channels"]:
        error("events_today", "fewer_events_than_observed_channels")
    if (facts["events_today"] > 0) != (facts["observed_channels"] > 0):
        error("events_today", "inconsistent_with_observed_channels")
    if facts["current_alarm"] != (facts["alarm_channels"] > 0):
        error("current_alarm", "inconsistent_with_alarm_channels")
    if facts["alarm_days_7d"] > facts["observed_days_7d"]:
        error("alarm_days_7d", "exceeds_observed_days")
    if facts["current_alarm"] and facts["alarm_days_7d"] == 0:
        error("alarm_days_7d", "omits_current_alarm_day")
    if facts["observed_channels"] > 0 and facts["observed_days_7d"] == 0:
        error("observed_days_7d", "omits_current_observed_day")
    if facts["observed_channels"] == 0 and facts["observed_days_7d"] == 7:
        error("observed_days_7d", "includes_unobserved_current_day")
    if not facts["current_alarm"] and facts["alarm_days_7d"] == 7:
        error("alarm_days_7d", "includes_unalarmed_current_day")
    return errors


def _identity_facts(facts: dict) -> dict:
    """Hash accepted scalars; describe unsupported types without rendering them."""
    result = {}
    for field in sorted(facts):
        value = facts[field]
        if type(value) in {str, int, bool} or value is None:
            result[field] = value
        else:
            # Unsupported values carry no factual evidence and cannot reach text.
            result[field] = {"invalid_type": type(value).__name__}
    return result


def recommend(facts: Mapping[str, object], context: Mapping[str, object]) -> dict:
    """Return review suggestions, never change a card, queue, or equipment state.

    Unknown fields and invalid release context raise RecommendationValidationError.
    Missing/invalid known facts return insufficient_data with no suggested steps.
    The result is JSON-compatible and deterministic; generated_at belongs outside
    this function. Its recommendation ID is not proof of input authenticity.
    """
    context = _context(context)
    facts = _project(facts, FACT_FIELDS | OPTIONAL_FACT_FIELDS, "facts")
    errors = _fact_errors(facts)
    result = {"schema_version": SCHEMA_VERSION, "ruleset_version": RULESET_VERSION,
        "ruleset_sha256": RULESET_SHA256, "status": "insufficient_data" if errors else "ok",
        **context, "fact_cutoff": context["feature_cutoff"],
        "recommendation_id": _sha({"context": context, "facts": _identity_facts(facts),
                                   "ruleset_sha256": RULESET_SHA256}),
        "rule_ids": [], "evidence": [], "steps": [], "validation_errors": errors,
        "requires_dispatcher_decision": True, "external_send": False,
        "limitations": list(_LIMITATIONS)}
    if errors:
        return result

    if facts["observed_channels"] == 0:
        selected = [0]
    else:
        selected = [i for i, condition in (
            (1, facts["observed_channels"] < facts["catalog_channels"]),
            (2, facts["current_alarm"]), (3, facts["alarm_days_7d"] >= 2)) if condition]
        if not selected:
            selected = [4 if facts["warning"] else 5]
    for index in selected[:3]:
        rule_id, _, evidence_fields, template = _RULES[index]
        result["rule_ids"].append(rule_id)
        result["steps"].append({"rule_id": rule_id, "text": template.format(**facts),
                                "evidence_fields": list(evidence_fields)})
    # Store all accepted facts, including the queue decision/rank used only for
    # identity; rule-specific evidence_fields identify actual supporting facts.
    for field in sorted(facts):
        evidence = {"field": field, "value": facts[field]}
        if field in {"warning", "rank"}:
            evidence.update(source="forecast_policy", available_at=context["issue_time"])
        else:
            evidence.update(source="observation", window_end=context["feature_cutoff"])
        result["evidence"].append(evidence)
    return result
