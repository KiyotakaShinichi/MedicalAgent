"""The RAG failure diagnostics must be safe, total, and log-publishable.

These helpers only run while building an assertion message, which is exactly
when they are hardest to debug: if one of them raised on the Linux runner, the
test would report *that* exception instead of the retrieval failure it exists
to explain, and the CI log would be less useful than before.

So the contract is:

* never raise, whatever shape the result has — including empty, partial, or
  missing telemetry;
* never emit patient identifiers, query text, reply text, document bodies, or
  credentials, because the output lands in a public Actions log;
* stay small and stably ordered, so a Linux payload can be diffed against a
  Windows one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.breast_monitoring.support import (  # noqa: E402
    _format_diagnostics,
    _rag_pipeline_diagnostics,
    _regression_failure_diagnostics,
)

# Values that must never reach a public log if they appear in a result.
_SECRETS = (
    "PT-000123",
    "I have fever during chemo",
    "Please call your oncology team",
    "sk-live-abcdef",
)

_RESULT = {
    "intent": "safety_boundary",
    "rag_mode": "urgent_safety_rag",
    "safety": {"level": "high_risk"},
    "reply": "Please call your oncology team.",
    "citations": [],
    "retrieval_context": [],
    "patient_id": "PT-000123",
    "query": "I have fever during chemo",
    "pipeline_trace": {
        "terminal_step": "generated",
        "retrieved_count": 40,
        "reranked_count": 4,
        "compressed_count": 0,
    },
    "retrieval_confidence": {"top_score": 1.2685, "top_k_evaluated": 1},
    "pregen_tier_filter": {
        "initial_retrieval": {"kept_count": 3, "dropped_count": 37, "kept_chunk_ids": ["a", "b"]}
    },
    "tier_filter": {"kept_count": 0, "dropped_count": 3, "kept_chunk_ids": []},
}


def test_pipeline_diagnostics_capture_the_retrieval_funnel() -> None:
    """The funnel is the whole point: it localises which stage returned zero."""
    payload = _rag_pipeline_diagnostics(_RESULT)
    for stage in (
        "retrieved_count", "reranked_count", "compressed_count",
        "pregen_tier_kept", "pregen_tier_dropped", "tier_kept", "tier_dropped",
        "retrieval_context_count",
    ):
        assert stage in payload, f"{stage} missing; the funnel would be unreadable"
    assert payload["retrieved_count"] == 40
    assert payload["compressed_count"] == 0
    assert payload["tier_kept"] == 0


def test_pipeline_diagnostics_record_the_environment() -> None:
    """Platform and Python identify a Linux-only divergence at a glance."""
    payload = _rag_pipeline_diagnostics(_RESULT)
    assert payload["platform"]
    assert payload["python"]
    assert "encoder_status" in payload


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"pipeline_trace": None, "tier_filter": None},
        {"retrieval_context": None, "citations": None},
        {"pipeline_trace": {}, "retrieval_confidence": {}, "pregen_tier_filter": {}},
    ],
)
def test_pipeline_diagnostics_never_raise_on_degraded_results(result: dict) -> None:
    """A diagnostic that raises would replace the real failure with its own."""
    assert isinstance(_rag_pipeline_diagnostics(result), dict)


def test_diagnostics_do_not_leak_identifiers_or_free_text() -> None:
    rendered = _format_diagnostics("probe", _rag_pipeline_diagnostics(_RESULT))
    for secret in _SECRETS:
        assert secret not in rendered, f"{secret!r} would be published to the CI log"


def test_rendered_diagnostics_are_stable_and_small() -> None:
    """Sorted keys so a Linux payload diffs cleanly against a Windows one."""
    rendered = _format_diagnostics("probe", _rag_pipeline_diagnostics(_RESULT))
    body = json.loads(rendered.split(":\n", 1)[1])
    assert list(body) == sorted(body), "unsorted keys make cross-platform diffing noisy"
    assert len(rendered) < 4000, "diagnostics must stay readable in an Actions log"


# ─── regression-suite diagnostics ────────────────────────────────────────────

_REPORT = {
    "case_count": 8,
    "summary": {
        "status": "unideal",
        "pass_rate": 0.75,
        "attack_block_rate": 1.0,
        "output_guardrail_pass_rate": 1.0,
        "expected_source_hit_rate": 0.8,
    },
    "cases": [
        {
            "id": "case-ok", "category": "education", "status": "passed",
            "checks": [{"name": "expected_source_hit", "passed": True}],
            "observed": {"intent": "education"},
        },
        {
            "id": "case-bad", "category": "education", "status": "failed",
            # The real report emits a *list* of {"name", "passed"} records.
            # An earlier draft assumed a mapping and raised while building the
            # failure message, masking the assertion it existed to explain.
            "checks": [
                {"name": "expected_source_hit", "passed": False},
                {"name": "output_guardrail", "passed": True},
            ],
            "observed": {
                "intent": "education",
                "retrieval_context_ids": ["s1", "s2"],
                "citation_ids": [],
                "grounding_score": 0.31,
                "patient_id": "PT-000123",
            },
        },
    ],
}


def test_regression_diagnostics_name_the_failing_cases() -> None:
    """`pass_rate < 0.80` is already known; *which* cases failed is not."""
    payload = _regression_failure_diagnostics(_REPORT)
    assert payload["failed_case_count"] == 1
    failed = payload["failed_cases"][0]
    assert failed["id"] == "case-bad"
    assert failed["failed_checks"] == ["expected_source_hit"], "must isolate the failing check"
    assert failed["retrieval_context_ids"] == ["s1", "s2"]
    assert payload["pass_rate"] == 0.75


def test_regression_diagnostics_do_not_leak_patient_identifiers() -> None:
    rendered = _format_diagnostics("probe", _regression_failure_diagnostics(_REPORT))
    assert "PT-000123" not in rendered


@pytest.mark.parametrize(
    "report",
    [
        {},
        {"summary": None, "cases": None},
        {"cases": []},
        # Wrong *types*, not just missing keys — the shape that actually broke
        # this helper. Empty/None fixtures alone did not catch it.
        {"cases": [{"id": "x", "status": "failed", "checks": {"a": False}}]},
        {"cases": [{"id": "x", "status": "failed", "checks": "unexpected"}]},
        {"cases": [{"id": "x", "status": "failed", "checks": None}]},
        {"cases": [{"id": "x", "status": "failed", "observed": "unexpected"}]},
    ],
)
def test_regression_diagnostics_never_raise(report: dict) -> None:
    assert isinstance(_regression_failure_diagnostics(report), dict)


def test_failed_check_names_handles_both_report_shapes() -> None:
    """List-of-records is what the suite emits; mapping is tolerated."""
    from tests.breast_monitoring.support import _failed_check_names

    assert _failed_check_names(
        [{"name": "b", "passed": False}, {"name": "a", "passed": False},
         {"name": "c", "passed": True}]
    ) == ["a", "b"]
    assert _failed_check_names({"z": False, "y": True}) == ["z"]
    assert _failed_check_names("unexpected") == []
    assert _failed_check_names(None) == []
