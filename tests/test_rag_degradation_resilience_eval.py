from backend.services.rag_degradation_resilience_eval import (
    build_rag_degradation_resilience_eval,
)


def test_degradation_drill_is_complete_and_nonclinical(tmp_path) -> None:
    payload = build_rag_degradation_resilience_eval(tmp_path / "result.json")
    assert payload["status"] == "strong_offline_drill"
    assert payload["case_count"] == 5
    assert payload["passed_count"] == 5
    assert payload["patient_agent_invoked"] is False
    assert payload["managed_network_request_performed"] is False
    assert payload["retrieval_improvement_proven"] is False
    assert payload["production_outage_recovery_proven"] is False
    assert payload["clinical_validation"] is False


def test_degradation_drill_has_expected_cases(tmp_path) -> None:
    payload = build_rag_degradation_resilience_eval(tmp_path / "result.json")
    ids = {case["case_id"] for case in payload["cases"]}
    assert ids == {
        "corrupted_index_rebuild",
        "stale_fingerprint_rebuild",
        "dense_component_unavailable_sparse_fallback",
        "missing_optional_metadata_tolerated",
        "empty_query_fails_closed",
    }
