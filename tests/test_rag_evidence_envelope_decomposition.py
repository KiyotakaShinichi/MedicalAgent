"""Compatibility contracts for the decomposed RAG evidence boundary."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import backend.services.rag_evidence_envelope as facade


EXPECTED_EXPORTS = [
    "AuthorizationDecision",
    "EVIDENCE_ENVELOPE_VERSION",
    "EVIDENCE_POLICY_VERSION",
    "EvidenceDisposition",
    "EvidenceEnvelope",
    "SAFETY_POLICY_VERSION",
    "VALIDATOR_POLICY_VERSION",
    "authorize_evidence_release",
    "build_evidence_envelope",
    "build_fail_closed_error_result",
    "build_safe_abstention",
    "enforce_evidence_release",
    "enforce_transport_release",
    "parse_evidence_envelope",
    "response_digest",
    "record_rag_cache_rejection",
    "snapshot_evidence_release_metrics",
    "validate_cached_response",
]

EXPECTED_SIGNATURES = {
    "authorize_evidence_release": (
        "(envelope: 'EvidenceEnvelope | Mapping[str, Any] | None') -> 'AuthorizationDecision'"
    ),
    "build_evidence_envelope": (
        "(result: 'Mapping[str, Any]', *, query: 'str' = '', "
        "retrieved: 'Sequence[Mapping[str, Any]] | None' = None, "
        "input_guardrails: 'Mapping[str, Any] | None' = None, request_id: 'str | None' = None, "
        "validation_errors: 'Sequence[str] | None' = None, "
        "validation_warnings: 'Sequence[str] | None' = None, "
        "evidence_required: 'bool | None' = None) -> 'EvidenceEnvelope'"
    ),
    "build_fail_closed_error_result": (
        "(*, query: 'str' = '', request_id: 'str | None' = None, "
        "error_code: 'str' = 'internal_pipeline_failure', "
        "result: 'MutableMapping[str, Any] | None' = None) -> 'MutableMapping[str, Any]'"
    ),
    "build_safe_abstention": (
        "(disposition: 'EvidenceDisposition', *, query: 'str' = '', "
        "input_guardrails: 'Mapping[str, Any] | None' = None, "
        "existing_reply: 'str' = '') -> 'str'"
    ),
    "enforce_evidence_release": (
        "(result: 'MutableMapping[str, Any]', *, query: 'str' = '', "
        "retrieved: 'Sequence[Mapping[str, Any]] | None' = None, "
        "input_guardrails: 'Mapping[str, Any] | None' = None, request_id: 'str | None' = None, "
        "validation_errors: 'Sequence[str] | None' = None, "
        "validation_warnings: 'Sequence[str] | None' = None, "
        "evidence_required: 'bool | None' = None) -> 'MutableMapping[str, Any]'"
    ),
    "enforce_transport_release": (
        "(result: 'MutableMapping[str, Any]', *, query: 'str' = '') -> 'MutableMapping[str, Any]'"
    ),
    "parse_evidence_envelope": (
        "(value: 'EvidenceEnvelope | Mapping[str, Any] | None') -> "
        "'tuple[EvidenceEnvelope | None, str | None]'"
    ),
    "record_rag_cache_rejection": "() -> 'None'",
    "response_digest": "(reply: 'Any') -> 'str'",
    "snapshot_evidence_release_metrics": "() -> 'dict[str, int]'",
    "validate_cached_response": (
        "(response: 'Mapping[str, Any] | None', *, "
        "policy: 'Mapping[str, Any] | None' = None) -> 'tuple[bool, str]'"
    ),
    "_high_risk_semantic_validation_required": (
        "(claims: 'Sequence[Mapping[str, Any]]') -> 'bool'"
    ),
}


def _valid_result() -> dict:
    chunk = {
        "id": "chunk-1",
        "source_id": "source-1",
        "source_tier": "T1",
        "allowed_use": ["patient_education"],
        "staleness_status": "current",
        "title": "Patient education source",
        "source_name": "Authoritative source",
        "source_url": "https://example.invalid/source-1",
    }
    return {
        "reply": "This is a source-backed educational statement.",
        "intent": "education",
        "safety": {"level": "low_risk", "scope": "allowed", "cache_allowed": True},
        "citations": [{"id": "chunk-1", "source_id": "source-1"}],
        "retrieval_context": [chunk],
        "rag_mode": "education_rag",
        "tier_filter": {
            "kept_chunk_ids": ["chunk-1"],
            "decisions": [{
                "chunk_id": "chunk-1",
                "source_id": "source-1",
                "tier": "T1",
                "allowed_use": ["patient_education"],
                "staleness_status": "current",
                "decision": "kept",
            }],
        },
        "claim_validation": {
            "citation_status": "complete",
            "claim_count": 1,
            "supported_count": 1,
            "weakly_supported_count": 0,
            "unsupported_count": 0,
            "verdicts": [{
                "sentence": "This is a source-backed educational statement.",
                "is_claim": True,
                "claim_type": "education",
                "status": "supported",
                "support_score": 0.96,
                "validation_method": "test_fixture",
                "supporting_chunk_ids": ["chunk-1"],
            }],
        },
        "retrieval_confidence": {
            "answerability_status": "answerable_with_citations",
            "evidence_conflict_flag": False,
        },
        "validation": {"status": "passed", "issues": []},
        "post_gen_validator": {"decision": "allowed", "answer_tier_escalation": None},
        "guardrails": {
            "input": {"status": "passed", "level": "low_risk"},
            "output": {"status": "passed", "issues": []},
        },
        "pipeline_trace": {"terminal_step": "generated"},
    }


def test_facade_preserves_exports_signatures_and_private_policy_hook():
    assert facade.__all__ == EXPECTED_EXPORTS
    for name, expected in EXPECTED_SIGNATURES.items():
        assert str(inspect.signature(getattr(facade, name))) == expected


def test_facade_preserves_valid_authorization_and_serialization_contract():
    result = _valid_result()
    authorized = facade.enforce_evidence_release(
        result,
        query="What does this educational term mean?",
        input_guardrails=result["guardrails"]["input"],
        request_id="decomposition-test",
    )
    envelope = authorized["evidence_envelope"]
    parsed, error = facade.parse_evidence_envelope(envelope)

    assert error is None
    assert parsed is not None
    assert parsed.to_dict() == envelope
    assert envelope["final_disposition"] == "ALLOW"
    assert envelope["source_ids"] == ["source-1"]
    assert authorized["release_authorization"] == {
        "disposition": "ALLOW",
        "release_evidence_answer": True,
        "release_safe_response": False,
        "reason": "validated_allow",
    }


def test_facade_transport_uses_patchable_classifier_and_fails_closed(monkeypatch):
    called: list[str] = []
    blocked = SimpleNamespace(
        blocked=True,
        reason="decomposition_fault_injection",
        to_dict=lambda: {"decision": "blocked", "reason": "decomposition_fault_injection"},
    )

    def classify(reply: str):
        called.append(reply)
        return blocked

    monkeypatch.setattr(facade, "classify_output_actionability", classify)
    result = facade.enforce_transport_release(
        {"reply": "candidate", "citations": [{"source_id": "source-1"}]},
        query="test",
    )

    assert called == ["candidate"]
    assert result["citations"] == []
    assert result["release_authorization"]["release_evidence_answer"] is False
    assert result["evidence_envelope"]["abstention_reason"] == (
        "semantic_output_guard:decomposition_fault_injection"
    )


def test_fail_closed_result_preserves_closed_schema_and_discards_candidate():
    result = facade.build_fail_closed_error_result(
        query="private patient query",
        request_id="decomposition-test",
        error_code="simulated_failure",
        result={"reply": "candidate", "citations": [{"source_id": "source-1"}]},
    )

    assert result["reply"] != "candidate"
    assert result["citations"] == []
    assert result["evidence_envelope"]["final_disposition"] == "INTERNAL_ERROR"
    assert result["evidence_envelope"]["clinical_validation"] is False
    assert result["evidence_envelope"]["validation_errors"] == ["simulated_failure"]
