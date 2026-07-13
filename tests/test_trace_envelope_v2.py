from __future__ import annotations

from backend.services.trace_envelope_v2 import (
    FORBIDDEN_EXACT_KEYS,
    REQUIRED_FIELDS,
    build_trace_envelope_v2,
    build_trace_envelope_v2_eval,
    validate_trace_envelope_v2,
)


def test_trace_envelope_v2_contains_required_redacted_fields():
    trace = build_trace_envelope_v2(
        {
            "intent": "education",
            "safety": {"level": "low", "scope": "education", "matched_terms": []},
            "pipeline_trace": {"terminal_step": "generated"},
            "sources": [{"source_id": "curated-source"}],
            "rag_evaluation": {"claim_support_rate": 1.0, "citation_precision": 1.0},
        },
        patient_id="P001",
        route="patient_chat",
        latency_ms={"total": 123.0},
        correlation_id="req-123",
    )

    assert set(REQUIRED_FIELDS) <= set(trace)
    assert trace["patient_id_hash"] != "P001"
    assert "patient_id" not in trace
    assert trace["clinical_validation"] is False
    assert trace["retrieval_backend"] == "local_source_governed_rag"
    assert trace["source_ids"] == ["curated-source"]
    assert "not clinical validation" in trace["claim_boundary"].lower()
    ok, issues = validate_trace_envelope_v2(trace)
    assert ok, issues


def test_trace_envelope_v2_scrubs_forbidden_fields():
    trace = build_trace_envelope_v2(
        {
            "intent": "education",
            "patient_id": "P001",
            "private_chain_of_thought": "hidden",
            "pipeline_trace": {"terminal_step": "generated"},
        },
        patient_id="P001",
        route="patient_chat",
        latency_ms=10,
        correlation_id="req-456",
    )

    text = str(trace)
    for forbidden in FORBIDDEN_EXACT_KEYS:
        assert forbidden not in trace
    assert "hidden" not in text
    ok, issues = validate_trace_envelope_v2(trace)
    assert ok, issues


def test_trace_envelope_v2_validator_catches_poisoned_payload():
    trace = build_trace_envelope_v2(
        {"intent": "education", "pipeline_trace": {"terminal_step": "generated"}},
        patient_id="P001",
        route="patient_chat",
        latency_ms=10,
        correlation_id="req-789",
    )
    trace["raw_patient_identifier"] = "P001"
    trace["private_chain_of_thought"] = "hidden"

    ok, issues = validate_trace_envelope_v2(trace)

    assert not ok
    assert any("raw_patient_identifier" in issue for issue in issues)
    assert any("private_chain_of_thought" in issue for issue in issues)


def test_trace_envelope_v2_eval_artifact(tmp_path):
    report = build_trace_envelope_v2_eval(
        output_path=tmp_path / "trace_v2.json",
        doc_path=tmp_path / "trace_v2.md",
    )

    assert report["status"] == "strong"
    assert report["clinical_validation"] is False
    assert report["healthcare_production_ready"] is False
    assert report["validation_pass_rate"] == 1.0
    assert report["forbidden_field_catch_rate"] == 1.0
    assert report["raw_patient_identifier_stored"] is False
    assert report["private_chain_of_thought_stored"] is False
