"""Catalog loading and normalization for patient-agent regression cases."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def load_case_catalog(path: str, *, path_type: type, json_module: Any) -> dict:
    if not path:
        return {}
    catalog_path = path_type(path)
    if not catalog_path.exists():
        return {}
    try:
        return json_module.loads(catalog_path.read_text(encoding="utf-8"))
    except json_module.JSONDecodeError:
        return {}


def normalize_rag_cases(
    catalog: dict,
    defaults: dict,
    *,
    allow_no_citations_fn: Callable[[str | None, str | None], bool],
    default_guardrail_status_fn: Callable[[str | None], str],
    default_safety_level_fn: Callable[[str | None], str],
) -> list[dict]:
    cases = []
    for raw in catalog.get("cases") or []:
        case_id = raw.get("id")
        if not case_id:
            continue
        default_case = defaults.get(case_id) or {}
        expected_intent = raw.get("expected_intent")
        category = raw.get("category") or default_case.get("category") or "education"
        allow_no_citations = raw.get("allow_no_citations")
        if allow_no_citations is None:
            allow_no_citations = allow_no_citations_fn(category, expected_intent)
        cases.append({
            "id": case_id,
            "category": category,
            "query": raw.get("input") or default_case.get("query") or "",
            "fallback_response": raw.get("fallback_response") or default_case.get("fallback_response") or "I can help track this for review.",
            "expected_intent": expected_intent or default_case.get("expected_intent"),
            "expected_sources": raw.get("expected_sources") or default_case.get("expected_sources") or [],
            "expected_context_keywords": raw.get("expected_context_keywords") or default_case.get("expected_context_keywords") or [],
            "expected_input_guardrail": raw.get("expected_input_guardrail") or default_guardrail_status_fn(expected_intent),
            "expected_safety_level": raw.get("expected_safety_level") or default_safety_level_fn(expected_intent),
            "expected_reply_terms": raw.get("required_reply_terms") or default_case.get("expected_reply_terms") or [],
            "allow_no_citations": allow_no_citations,
            "should_block": expected_intent == "security_boundary",
            "expected_cacheable": raw.get("cacheable"),
        })
    return cases


def normalize_safety_cases(
    catalog: dict,
    defaults: dict,
    *,
    default_guardrail_status_fn: Callable[[str | None], str],
    default_safety_level_fn: Callable[[str | None], str],
) -> list[dict]:
    cases = []
    for raw in catalog.get("cases") or []:
        case_id = raw.get("automated_case_id") or raw.get("id")
        if not case_id:
            continue
        expected_intent = raw.get("expected_route") or raw.get("expected_intent")
        default_case = defaults.get(case_id) or {}
        cases.append({
            "id": case_id,
            "category": raw.get("category") or default_case.get("category") or "clinical_safety",
            "query": raw.get("input") or default_case.get("query") or "",
            "fallback_response": raw.get("fallback_response") or default_case.get("fallback_response") or "I can help track this for review.",
            "expected_intent": expected_intent or default_case.get("expected_intent"),
            "expected_sources": raw.get("expected_sources") or default_case.get("expected_sources") or [],
            "expected_context_keywords": raw.get("expected_context_keywords") or default_case.get("expected_context_keywords") or [],
            "expected_input_guardrail": raw.get("expected_guardrail_status") or default_guardrail_status_fn(expected_intent),
            "expected_safety_level": raw.get("expected_safety_level") or default_safety_level_fn(expected_intent),
            "expected_reply_terms": raw.get("required_reply_terms") or default_case.get("expected_reply_terms") or [],
            "allow_no_citations": True,
            "should_block": expected_intent == "security_boundary",
            "expected_cacheable": False,
        })
    return cases


def allow_no_citations(category: str | None, expected_intent: str | None) -> bool:
    if expected_intent in {"security_boundary", "safety_boundary", "treatment_decision_boundary"}:
        return True
    return category in {"conversation", "emotional_support", "general_support"}


def default_guardrail_status(expected_intent: str | None) -> str:
    if expected_intent == "security_boundary":
        return "failed"
    return "passed"


def default_safety_level(expected_intent: str | None) -> str:
    if expected_intent in {"security_boundary", "safety_boundary", "treatment_decision_boundary"}:
        return "high_risk"
    return "low_risk"


def dedupe_cases(cases: list[dict]) -> list[dict]:
    seen = set()
    output = []
    for case in cases:
        case_id = case.get("id")
        if not case_id or case_id in seen:
            continue
        seen.add(case_id)
        output.append(case)
    return output
