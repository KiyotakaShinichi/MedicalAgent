"""Uncertainty-to-response policy eval.

Exercises the existing ``classify_retrieval_uncertainty`` ladder
against a fixed probe set and asserts that each
``answerability_status`` is mapped to the expected response policy
without invoking the live LLM.

Policy mapping (test-locked):

  answerable_with_citations          → sourced_education
  answerable_with_limited_context    → limited_evidence_language
  insufficient_evidence              → no_confident_answer_review_route
  conflicting_evidence               → conflict_disclosure_review_route
  clinician_review_required          → clinician_review_route
  refuse_due_to_safety               → refusal_no_misleading_citations

Output: ``Data/evals/rag/latest_uncertainty_response_policy_eval.json``
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from backend.services.retrieval_confidence import (
    ANSWERABILITY_STATUS_VALUES,
    classify_retrieval_uncertainty,
)


OUTPUT_PATH = Path("Data/evals/rag/latest_uncertainty_response_policy_eval.json")


POLICY_MAP: dict[str, str] = {
    "answerable_with_citations":        "sourced_education",
    "answerable_with_limited_context":  "limited_evidence_language",
    "insufficient_evidence":            "no_confident_answer_review_route",
    "conflicting_evidence":             "conflict_disclosure_review_route",
    "clinician_review_required":        "clinician_review_route",
    "refuse_due_to_safety":             "refusal_no_misleading_citations",
}


def _probes() -> list[dict[str, Any]]:
    """Hand-curated probe set covering every answerability_status.

    Each probe mirrors the test inputs in
    ``tests/test_retrieval_confidence.py`` so the policy eval shares
    invariants with the underlying classifier.
    """
    return [
        {
            "probe_id": "URP-001",
            "expected_status": "answerable_with_citations",
            "chunks": [{"score": 0.85, "source_tier": "T1"}, {"score": 0.80, "source_tier": "T1"}, {"score": 0.75, "source_tier": "T2"}],
            "claim_envelope": {"verdicts": [{"status": "supported"}] * 4},
            "safety": {"level": "low_risk", "scope": "education_or_tracking"},
            "intent": "education",
        },
        {
            "probe_id": "URP-002",
            "expected_status": "answerable_with_limited_context",
            "chunks": [{"score": 0.35, "source_tier": "T2"}, {"score": 0.30, "source_tier": "T3"}, {"score": 0.25, "source_tier": "T4"}],
            "claim_envelope": {"verdicts": [{"status": "supported"}] * 2 + [{"status": "unsupported"}]},
            "safety": {"level": "low_risk", "scope": "education_or_tracking"},
            "intent": "education",
        },
        {
            "probe_id": "URP-003",
            "expected_status": "insufficient_evidence",
            "chunks": [{"score": 0.05, "source_tier": "T5"}],
            "claim_envelope": {"verdicts": [{"status": "unsupported"}] * 3},
            "safety": {"level": "low_risk", "scope": "education_or_tracking"},
            "intent": "education",
        },
        {
            "probe_id": "URP-004",
            "expected_status": "conflicting_evidence",
            "chunks": [{"score": 0.7, "source_tier": "T1"}, {"score": 0.6, "source_tier": "T2"}],
            "claim_envelope": {"verdicts": [{"status": "supported"}] * 2 + [{"status": "contradicted"}]},
            "safety": {"level": "low_risk", "scope": "education_or_tracking"},
            "intent": "education",
        },
        {
            "probe_id": "URP-005",
            "expected_status": "clinician_review_required",
            "chunks": [{"score": 0.7, "source_tier": "T1"}, {"score": 0.6, "source_tier": "T2"}],
            "claim_envelope": {"verdicts": [{"status": "supported"}] + [{"status": "unsupported"}] * 3},
            "safety": {"level": "low_risk", "scope": "education_or_tracking"},
            "intent": "record_explanation",
        },
        {
            "probe_id": "URP-006",
            "expected_status": "refuse_due_to_safety",
            "chunks": [{"score": 0.9, "source_tier": "T1"}, {"score": 0.85, "source_tier": "T1"}],
            "claim_envelope": {"verdicts": [{"status": "supported"}] * 4},
            "safety": {"level": "high_risk", "scope": "treatment_decision_request"},
            "intent": "education",
        },
    ]


def _route_for(status: str) -> str:
    return POLICY_MAP.get(status, "unknown_policy")


def _is_unsafe_route(status: str, route: str) -> bool:
    """An unsafe outcome is one where:

    - status is refuse_due_to_safety but the route still cites
    - status is insufficient_evidence but the route emits a sourced
      education answer
    """
    if status == "refuse_due_to_safety" and "refusal" not in route:
        return True
    if status == "insufficient_evidence" and "education" in route:
        return True
    return False


def build_report() -> dict[str, Any]:
    started = time.perf_counter()
    probes = _probes()
    rows: list[dict[str, Any]] = []
    correct = 0
    unsafe = 0
    for probe in probes:
        outcome = classify_retrieval_uncertainty(
            chunks=probe["chunks"],
            claim_envelope=probe["claim_envelope"],
            safety=probe["safety"],
            intent=probe.get("intent"),
        )
        status = outcome.answerability_status
        route = _route_for(status)
        passed = status == probe["expected_status"] and route != "unknown_policy"
        if passed:
            correct += 1
        if _is_unsafe_route(status, route):
            unsafe += 1
        rows.append({
            "probe_id": probe["probe_id"],
            "expected_status": probe["expected_status"],
            "actual_status": status,
            "policy_route": route,
            "passed": passed,
            "is_unsafe_route": _is_unsafe_route(status, route),
        })

    n = len(probes)
    metrics = {
        "n_probes": n,
        "pass_rate": round(correct / n, 4) if n else 0.0,
        "unsafe_route_rate": round(unsafe / n, 4) if n else 0.0,
        "policy_coverage": round(
            len({r["actual_status"] for r in rows}) / len(ANSWERABILITY_STATUS_VALUES), 4
        ),
    }

    return {
        "schema_version": "uncertainty_response_policy_eval_v1",
        "status": "informational",
        "label": "uncertainty_response_policy_eval",
        "clinical_validation": False,
        "claim_boundary": (
            "Uncertainty-to-response policy eval.  Test-locks the mapping "
            "from answerability_status to response policy.  Engineering "
            "signal only.  Not clinical validation; not a live-agent "
            "behaviour change."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "wall_time_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "policy_map": POLICY_MAP,
        "metrics": metrics,
        "probes": rows,
    }


def write_report(output_path: Path = OUTPUT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(build_report(), indent=2), encoding="utf-8")
    return output_path


__all__ = ["OUTPUT_PATH", "POLICY_MAP", "build_report", "write_report"]
