"""Compatibility and behavioral lock-ins for the agent regression facade."""

from __future__ import annotations

import hashlib
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base
from backend.models import Patient
from backend.services import agent_regression_eval as regression
from backend.services.agent_rag import run_patient_agent_pipeline
from backend.services.agent_regression_eval_utils import (
    numeric,
    rate,
    round_mean,
    source_ids,
    status_meaning,
)


EXPECTED_SURFACE = {
    "Base",
    "DEFAULT_AGENT_EVAL_CASES",
    "DEFAULT_AGENT_REGRESSION_PATH",
    "DEFAULT_RAG_CASES_PATH",
    "DEFAULT_SAFETY_CASES_PATH",
    "EVALS_DIR",
    "EVAL_PATIENT_ID",
    "Path",
    "Patient",
    "_allow_no_citations",
    "_check",
    "_check_rate",
    "_dedupe_cases",
    "_default_guardrail_status",
    "_default_safety_level",
    "_evaluate_case",
    "_load_case_catalog",
    "_normalize_rag_cases",
    "_normalize_safety_cases",
    "_numeric",
    "_overall_status",
    "_quality_gates",
    "_rate",
    "_round_mean",
    "_run_case",
    "_seed_eval_patient",
    "_source_ids",
    "_status_meaning",
    "_summary",
    "_temp_db_session",
    "create_engine",
    "datetime",
    "json",
    "load_eval_cases",
    "load_latest_agent_regression_report",
    "run_agent_regression_suite",
    "run_patient_agent_pipeline",
    "sessionmaker",
    "timezone",
}

EXPECTED_SIGNATURES = {
    "run_agent_regression_suite": "(output_path='Data/agent_eval/latest_agent_regression.json', cases=None)",
    "load_latest_agent_regression_report": "(path='Data/agent_eval/latest_agent_regression.json')",
    "_load_case_catalog": "(path: str) -> dict",
    "_normalize_rag_cases": "(catalog: dict, defaults: dict) -> list[dict]",
    "_normalize_safety_cases": "(catalog: dict, defaults: dict) -> list[dict]",
    "_allow_no_citations": "(category: str | None, expected_intent: str | None) -> bool",
    "_default_guardrail_status": "(expected_intent: str | None) -> str",
    "_default_safety_level": "(expected_intent: str | None) -> str",
    "_dedupe_cases": "(cases: list[dict]) -> list[dict]",
    "_run_case": "(db, index, case)",
    "_evaluate_case": "(case, result)",
    "_summary": "(results)",
    "_overall_status": "(metrics)",
    "_quality_gates": "()",
    "_seed_eval_patient": "(db)",
    "_temp_db_session": "()",
    "_check": "(name, passed, expected, observed)",
    "_check_rate": "(results, check_name)",
}

DEFAULT_CASES_SHA256 = "67fd4ff115124cf92cffa7a3f337d3514b2c5b233c7d67f3da334c0000232ae9"


def _representative_cases() -> list[dict]:
    return [
        {
            "id": "fixture-education",
            "category": "education",
            "query": "education",
            "fallback_response": "fallback",
            "expected_intent": "patient_education",
            "expected_sources": ["education-source"],
            "expected_input_guardrail": "passed",
            "expected_safety_level": "low_risk",
            "expected_reply_terms": ["education"],
            "allow_no_citations": False,
            "should_block": False,
            "expected_cacheable": True,
        },
        {
            "id": "fixture-security",
            "category": "security",
            "query": "security",
            "fallback_response": "fallback",
            "expected_intent": "security_boundary",
            "expected_sources": [],
            "expected_input_guardrail": "failed",
            "expected_safety_level": "high_risk",
            "expected_reply_terms": [],
            "allow_no_citations": True,
            "should_block": True,
            "expected_cacheable": False,
        },
    ]


def _fake_pipeline(**kwargs) -> dict:
    security = kwargs["query"] == "security"
    return {
        "intent": "security_boundary" if security else "patient_education",
        "safety": {"level": "high_risk" if security else "low_risk"},
        "guardrails": {
            "input": {"status": "failed" if security else "passed"},
            "output": {"status": "passed"},
        },
        "cache": {
            "status": "blocked_by_input_guardrail" if security else "miss",
            "cacheable": not security,
        },
        "citations": [] if security else [{"id": "education-source"}],
        "retrieval_context": (
            [] if security else [{"id": "education-source", "text": "education"}]
        ),
        "reply": "blocked" if security else "education",
        "rag_evaluation": {
            "answer_grounding": {"score": 0.9},
            "hallucination": {"score": 0.1, "risk": "low"},
            "cost_latency": {"latency_ms": 12.5, "estimated_total_tokens": 42},
        },
    }


def test_facade_preserves_committed_runtime_surface_and_imported_attributes() -> None:
    observed = {name for name in dir(regression) if not name.startswith("__")}
    assert observed == EXPECTED_SURFACE
    assert "__all__" not in regression.__dict__

    assert regression.json is json
    assert regression.datetime is datetime
    assert regression.timezone is timezone
    assert regression.Path is Path
    assert regression.create_engine is create_engine
    assert regression.sessionmaker is sessionmaker
    assert regression.Base is Base
    assert regression.Patient is Patient
    assert regression.run_patient_agent_pipeline is run_patient_agent_pipeline
    assert regression._numeric is numeric
    assert regression._rate is rate
    assert regression._round_mean is round_mean
    assert regression._source_ids is source_ids
    assert regression._status_meaning is status_meaning


def test_facade_preserves_committed_callable_signatures() -> None:
    expected = {
        **EXPECTED_SIGNATURES,
        "load_eval_cases": (
            f"(rag_cases_path: str = {regression.DEFAULT_RAG_CASES_PATH!r}, "
            f"safety_cases_path: str = {regression.DEFAULT_SAFETY_CASES_PATH!r})"
        ),
    }
    observed = {
        name: str(inspect.signature(getattr(regression, name)))
        for name in expected
    }
    assert observed == expected


def test_default_case_catalog_preserves_content_and_order() -> None:
    encoded = json.dumps(
        regression.DEFAULT_AGENT_EVAL_CASES,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    assert len(regression.DEFAULT_AGENT_EVAL_CASES) == 45
    assert hashlib.sha256(encoded).hexdigest() == DEFAULT_CASES_SHA256


def test_suite_preserves_schema_metrics_order_and_pipeline_monkeypatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = []

    def fake_pipeline(**kwargs) -> dict:
        calls.append(kwargs["query"])
        return _fake_pipeline(**kwargs)

    monkeypatch.setattr(regression, "run_patient_agent_pipeline", fake_pipeline)
    output_path = tmp_path / "agent-regression.json"
    payload = regression.run_agent_regression_suite(
        output_path=str(output_path),
        cases=_representative_cases(),
    )

    assert calls == ["education", "security"]
    assert output_path.exists()
    assert list(payload) == [
        "schema_version",
        "generated_at",
        "purpose",
        "case_catalogs",
        "case_count",
        "summary",
        "quality_gates",
        "cases",
        "limitations",
    ]
    assert list(payload["summary"]) == [
        "pass_rate",
        "intent_accuracy",
        "expected_source_hit_rate",
        "citation_presence_rate",
        "cache_policy_accuracy",
        "attack_block_rate",
        "output_guardrail_pass_rate",
        "average_grounding_score",
        "average_hallucination_score",
        "average_latency_ms",
        "average_estimated_total_tokens",
        "status",
        "meaning",
    ]
    assert payload["summary"] == {
        "pass_rate": 1.0,
        "intent_accuracy": 1.0,
        "expected_source_hit_rate": 1.0,
        "citation_presence_rate": 1.0,
        "cache_policy_accuracy": 1.0,
        "attack_block_rate": 1.0,
        "output_guardrail_pass_rate": 1.0,
        "average_grounding_score": 0.9,
        "average_hallucination_score": 0.1,
        "average_latency_ms": 12.5,
        "average_estimated_total_tokens": 42.0,
        "status": "strong",
        "meaning": "All current regression gates passed with good quality proxies.",
    }
    assert [row["id"] for row in payload["cases"]] == [
        "fixture-education",
        "fixture-security",
    ]
    assert [gate["minimum"] for gate in payload["quality_gates"]] == [
        1.0,
        1.0,
        0.85,
        0.8,
        0.9,
        0.9,
    ]


def test_internal_facade_monkeypatch_seams_drive_extracted_implementations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = []
    original = regression._check

    def recording_check(name, passed, expected, observed):
        seen.append(name)
        return original(name, passed, expected, observed)

    monkeypatch.setattr(regression, "_check", recording_check)
    regression._evaluate_case(_representative_cases()[0], _fake_pipeline(query="education"))
    assert seen == [
        "intent",
        "input_guardrail",
        "output_guardrail",
        "safety_level",
        "expected_source_hit",
        "reply_terms",
        "has_citations",
        "cacheable_policy",
    ]


def test_catalog_fallback_and_error_behavior_are_preserved(tmp_path: Path) -> None:
    missing = regression.load_latest_agent_regression_report(tmp_path / "missing.json")
    assert missing == {
        "status": "unavailable",
        "message": "No agent regression report has been generated yet.",
        "path": str(tmp_path / "missing.json"),
        "quality_gates": regression._quality_gates(),
    }

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{not-json", encoding="utf-8")
    assert regression._load_case_catalog(str(invalid)) == {}
    assert regression.load_eval_cases(
        str(tmp_path / "missing-rag.json"),
        str(tmp_path / "missing-safety.json"),
    ) is regression.DEFAULT_AGENT_EVAL_CASES
