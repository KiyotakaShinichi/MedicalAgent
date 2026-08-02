from __future__ import annotations

from backend.services.admin_benchmark_response import get_normalized_benchmark_artifact


REQUIRED_KEYS = {
    "status",
    "headline_metric",
    "metrics",
    "rows",
    "artifact_path",
    "last_run_at",
    "claim_boundary",
    "can_rerun",
    "errors",
}


def test_normalized_benchmark_artifact_has_stable_shape() -> None:
    payload = get_normalized_benchmark_artifact("live_rag_eval")

    assert REQUIRED_KEYS.issubset(payload)
    assert isinstance(payload["metrics"], dict)
    assert isinstance(payload["rows"], list)
    assert payload["artifact_path"] == "Data/evals/rag/latest_live_rag_eval.json"
    assert "clinical" in payload["claim_boundary"].lower()


def test_unknown_benchmark_artifact_returns_missing_envelope() -> None:
    payload = get_normalized_benchmark_artifact("does_not_exist")

    assert REQUIRED_KEYS.issubset(payload)
    assert payload["status"] == "missing"
    assert payload["can_rerun"] is False
    assert payload["errors"]


def test_research_paper_query_telemetry_is_exposed_with_query_rows() -> None:
    payload = get_normalized_benchmark_artifact("research_paper_query_telemetry")

    assert REQUIRED_KEYS.issubset(payload)
    assert payload["artifact_path"] == "Data/evals/rag/latest_research_paper_query_telemetry.json"
    if payload["status"] != "missing":
        assert payload["metrics"]["query_count"] >= 1
        assert isinstance(payload["rows"], list)
