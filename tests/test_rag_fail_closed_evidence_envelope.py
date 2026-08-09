from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from backend.services.rag_evidence_envelope import (
    EVIDENCE_ENVELOPE_VERSION,
    EVIDENCE_POLICY_VERSION,
    SAFETY_POLICY_VERSION,
    VALIDATOR_POLICY_VERSION,
    EvidenceDisposition,
    authorize_evidence_release,
    enforce_evidence_release,
    enforce_transport_release,
    parse_evidence_envelope,
    response_digest,
    validate_cached_response,
)


def _valid_evidence_result() -> dict:
    chunk = {
        "id": "chunk-1",
        "source_id": "source-1",
        "source_tier": "T1",
        "allowed_use": ["patient_education"],
        "staleness_status": "current",
        "title": "Patient education source",
        "source_name": "Authoritative source",
        "source_url": "https://example.invalid/source-1",
        "text": "A source-backed educational statement for testing.",
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


def _authorize_valid_result() -> dict:
    result = _valid_evidence_result()
    return enforce_evidence_release(
        result,
        query="What does this educational term mean?",
        input_guardrails=result["guardrails"]["input"],
    )


FAULT_CASES = (
    "retriever_exception",
    "reranker_exception",
    "query_rewriter_exception",
    "parent_expansion_exception",
    "source_filter_exception",
    "source_metadata_parser_exception",
    "provider_exception",
    "generation_exception",
    "malformed_generation",
    "truncated_output",
    "claim_extraction_exception",
    "claim_validator_timeout",
    "citation_validator_exception",
    "evidence_grader_exception",
    "uncertainty_classifier_exception",
    "telemetry_exception",
    "trace_exception",
    "persistence_exception",
    "empty_retrieval",
    "malformed_retrieval",
    "unknown_staleness",
    "missing_source_id",
    "missing_source_tier",
    "missing_answerability",
    "conflicting_evidence",
    "limited_context",
    "insufficient_evidence",
    "clinician_review_required",
    "safety_refusal",
    "partial_citations",
    "unsupported_citations",
    "unsupported_claim",
    "weak_claim",
    "missing_claims",
    "output_guardrail_failed",
    "post_gen_unavailable",
    "base_validation_failed",
    "post_gen_blocked",
    "input_guardrail_failed",
    "privacy_boundary",
    "research_evidence_abstention",
    "multiple_simultaneous_failures",
)


def _inject_fault(result: dict, case: str) -> list[str]:
    component_errors = {
        "retriever_exception",
        "reranker_exception",
        "query_rewriter_exception",
        "parent_expansion_exception",
        "source_filter_exception",
        "source_metadata_parser_exception",
        "provider_exception",
        "generation_exception",
        "malformed_generation",
        "truncated_output",
        "claim_extraction_exception",
        "claim_validator_timeout",
        "citation_validator_exception",
        "evidence_grader_exception",
        "uncertainty_classifier_exception",
        "telemetry_exception",
        "trace_exception",
        "persistence_exception",
    }
    if case in component_errors:
        return [case]
    if case == "empty_retrieval":
        result["retrieval_context"] = []
    elif case == "malformed_retrieval":
        result["retrieval_context"] = {"unexpected": "mapping"}
    elif case == "unknown_staleness":
        result["retrieval_context"][0]["staleness_status"] = "unknown"
        result["tier_filter"]["decisions"][0]["staleness_status"] = "unknown"
    elif case == "missing_source_id":
        result["retrieval_context"][0]["source_id"] = ""
        result["tier_filter"]["decisions"][0]["source_id"] = ""
    elif case == "missing_source_tier":
        result["retrieval_context"][0]["source_tier"] = ""
        result["tier_filter"]["decisions"][0]["tier"] = ""
    elif case == "missing_answerability":
        result["retrieval_confidence"].pop("answerability_status")
    elif case == "conflicting_evidence":
        result["retrieval_confidence"].update({
            "answerability_status": "conflicting_evidence",
            "evidence_conflict_flag": True,
        })
    elif case in {"limited_context", "insufficient_evidence", "clinician_review_required"}:
        mapping = {
            "limited_context": "answerable_with_limited_context",
            "insufficient_evidence": "insufficient_evidence",
            "clinician_review_required": "clinician_review_required",
        }
        result["retrieval_confidence"]["answerability_status"] = mapping[case]
    elif case == "safety_refusal":
        result["retrieval_confidence"]["answerability_status"] = "refuse_due_to_safety"
    elif case in {"partial_citations", "unsupported_citations"}:
        result["claim_validation"]["citation_status"] = case.split("_")[0]
    elif case == "unsupported_claim":
        result["claim_validation"].update({"supported_count": 0, "unsupported_count": 1})
    elif case == "weak_claim":
        result["claim_validation"].update({"supported_count": 0, "weakly_supported_count": 1})
    elif case == "missing_claims":
        result["claim_validation"].update({
            "claim_count": 0,
            "supported_count": 0,
            "verdicts": [],
        })
    elif case == "output_guardrail_failed":
        result["guardrails"]["output"] = {"status": "failed", "issues": ["unsafe_output"]}
    elif case == "post_gen_unavailable":
        result["post_gen_validator"] = {"decision": "unavailable"}
    elif case == "base_validation_failed":
        result["validation"] = {"status": "failed", "issues": ["malformed"]}
    elif case == "post_gen_blocked":
        result["post_gen_validator"] = {"decision": "blocked"}
    elif case == "input_guardrail_failed":
        result["guardrails"]["input"] = {"status": "failed", "level": "blocked"}
    elif case == "privacy_boundary":
        result["safety"]["scope"] = "privacy_boundary"
    elif case == "research_evidence_abstention":
        result["research_evidence_answerability"] = {"requires_abstention": True}
    elif case == "multiple_simultaneous_failures":
        result["retrieval_context"] = []
        result["validation"] = {"status": "failed"}
        return ["provider_exception", "claim_validator_timeout"]
    return []


def test_valid_complete_envelope_is_the_only_positive_allow_case():
    result = _authorize_valid_result()
    assert result["release_authorization"]["disposition"] == EvidenceDisposition.ALLOW.value
    assert result["release_authorization"]["release_evidence_answer"] is True
    assert result["citations"]
    assert result["evidence_envelope"]["response_digest"] == response_digest(result["reply"])


@pytest.mark.parametrize("case", FAULT_CASES)
def test_fault_matrix_never_releases_candidate_evidence_answer(case):
    result = _valid_evidence_result()
    original = result["reply"]
    errors = _inject_fault(result, case)
    input_guardrails = result["guardrails"]["input"]
    enforce_evidence_release(
        result,
        query="Maaari mo ba itong ipaliwanag sa akin?",
        input_guardrails=input_guardrails,
        validation_errors=errors,
    )
    assert len(FAULT_CASES) >= 30
    assert result["release_authorization"]["disposition"] != EvidenceDisposition.ALLOW.value
    assert result["release_authorization"]["release_evidence_answer"] is False
    assert result["reply"] != original
    assert result["citations"] == []


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_envelope",
        "wrong_envelope_version",
        "wrong_policy_version",
        "wrong_safety_version",
        "wrong_validator_version",
        "unknown_disposition",
        "non_boolean_evidence_required",
        "malformed_collection",
        "missing_request_id",
    ),
)
def test_malformed_or_unknown_envelope_is_denied(mutation):
    envelope = copy.deepcopy(_authorize_valid_result()["evidence_envelope"])
    if mutation == "missing_envelope":
        envelope = None
    elif mutation == "wrong_envelope_version":
        envelope["version"] = "future"
    elif mutation == "wrong_policy_version":
        envelope["policy_version"] = "future"
    elif mutation == "wrong_safety_version":
        envelope["safety_policy_version"] = "future"
    elif mutation == "wrong_validator_version":
        envelope["validator_version"] = "future"
    elif mutation == "unknown_disposition":
        envelope["final_disposition"] = "MAYBE"
    elif mutation == "non_boolean_evidence_required":
        envelope["evidence_required"] = "false"
    elif mutation == "malformed_collection":
        envelope["evidence_items"] = "not-a-list"
    elif mutation == "missing_request_id":
        envelope["request_id"] = ""
    decision = authorize_evidence_release(envelope)
    assert decision.disposition != EvidenceDisposition.ALLOW
    assert decision.release_evidence_answer is False


def test_transport_rejects_reply_mutation_after_authorization():
    result = _authorize_valid_result()
    result["reply"] = "Mutated unsupported medical answer."
    enforce_transport_release(result, query="Explain this")
    assert result["release_authorization"]["disposition"] == EvidenceDisposition.INTERNAL_ERROR.value
    assert result["citations"] == []
    assert "Mutated unsupported" not in result["reply"]


def test_transport_rejects_alternate_nested_payload_without_envelope():
    result = {
        "reply": "Unvalidated nested answer",
        "agent_pipeline": {"reply": "Unvalidated nested answer", "citations": [{"id": "x"}]},
    }
    enforce_transport_release(result, query="What does this mean?")
    assert result["release_authorization"]["disposition"] == EvidenceDisposition.INTERNAL_ERROR.value
    assert result["citations"] == []
    assert "Unvalidated nested answer" not in result["reply"]


def test_cache_requires_current_allow_envelope_and_policy():
    result = _authorize_valid_result()
    policy = {
        "evidence_envelope_version": EVIDENCE_ENVELOPE_VERSION,
        "evidence_policy_version": EVIDENCE_POLICY_VERSION,
        "safety_policy_version": SAFETY_POLICY_VERSION,
        "validator_version": VALIDATOR_POLICY_VERSION,
    }
    assert validate_cached_response(result, policy=policy) == (True, "cache_envelope_valid")
    for key in tuple(policy):
        stale = dict(policy)
        stale[key] = "old"
        valid, reason = validate_cached_response(result, policy=stale)
        assert valid is False
        assert reason.startswith("cache_policy_mismatch:")


def test_cache_rejects_legacy_missing_or_mutated_envelope():
    policy = {
        "evidence_envelope_version": EVIDENCE_ENVELOPE_VERSION,
        "evidence_policy_version": EVIDENCE_POLICY_VERSION,
        "safety_policy_version": SAFETY_POLICY_VERSION,
        "validator_version": VALIDATOR_POLICY_VERSION,
    }
    assert validate_cached_response({"reply": "legacy"}, policy=policy)[0] is False
    result = _authorize_valid_result()
    result["reply"] = "changed cached text"
    assert validate_cached_response(result, policy=policy)[0] is False


def test_taglish_abstention_does_not_add_emergency_language_without_urgent_input():
    result = _valid_evidence_result()
    enforce_evidence_release(
        result,
        query="Pwede mo ba ipaliwanag kung ano ito?",
        input_guardrails=result["guardrails"]["input"],
        validation_errors=["provider_timeout"],
    )
    lower = result["reply"].lower()
    assert "hindi ko" in lower
    assert "emergency" not in lower
    assert "911" not in lower


def test_observability_events_do_not_store_raw_query_or_passages():
    sensitive_marker = "unique-patient-marker-92831"
    result = _valid_evidence_result()
    enforce_evidence_release(
        result,
        query=sensitive_marker,
        input_guardrails=result["guardrails"]["input"],
    )
    serialized = json.dumps(result["evidence_envelope_events"])
    assert sensitive_marker not in serialized
    assert result["retrieval_context"][0]["text"] not in serialized


def test_malformed_event_sink_still_returns_safe_internal_error():
    from backend.services.rag_evidence_envelope import build_fail_closed_error_result

    result = {"reply": "candidate", "evidence_envelope_events": "bad-sink"}
    build_fail_closed_error_result(query="test", result=result, error_code="boundary_fault")
    assert result["release_authorization"]["disposition"] == EvidenceDisposition.INTERNAL_ERROR.value
    assert isinstance(result["evidence_envelope_events"], list)
    assert result["citations"] == []


def test_patient_agent_top_level_exception_is_fail_closed(monkeypatch):
    from backend.services import agent_rag

    def explode(**_kwargs):
        raise TimeoutError("provider timed out with raw content")

    monkeypatch.setattr(agent_rag, "_run_patient_agent_pipeline_impl", explode)
    result = agent_rag.run_patient_agent_pipeline(None, "P001", "Explain CBC", {}, "fallback")
    assert result["release_authorization"]["disposition"] == EvidenceDisposition.INTERNAL_ERROR.value
    assert "raw content" not in json.dumps(result)


def test_intent_aware_governance_exception_replaces_candidate(monkeypatch):
    from backend.services.agent_post_gen import apply_intent_aware_rag_layer
    from backend.services import rag_claim_validator

    def explode(*_args, **_kwargs):
        raise TimeoutError("validator raw payload")

    monkeypatch.setattr(rag_claim_validator, "validate_claims", explode)
    result = _valid_evidence_result()
    original = result["reply"]
    decision = type("Decision", (), {"decision": "allowed"})()
    apply_intent_aware_rag_layer(result, result["retrieval_context"], {"status": "passed"}, decision)
    assert result["reply"] != original
    assert result["citations"] == []
    assert result["rag_governance_error"]["stage"] == "claim_validation"
    assert "validator raw payload" not in json.dumps(result)


@pytest.mark.parametrize(
    ("module_name", "attribute", "expected_stage"),
    (
        ("rag_intent_modes", "select_mode", "select_mode"),
        ("rag_tier_filter", "filter_chunks_by_mode", "source_tier_filter"),
        ("rag_claim_validator", "validate_claims", "claim_validation"),
        ("rag_evidence_grading", "grade_evidence", "evidence_grading"),
        ("retrieval_confidence", "classify_retrieval_uncertainty", "retrieval_uncertainty"),
    ),
)
def test_each_governance_dependency_exception_fails_closed(
    monkeypatch,
    module_name,
    attribute,
    expected_stage,
):
    from backend.services.agent_post_gen import apply_intent_aware_rag_layer
    from backend.services import (
        rag_claim_validator,
        rag_evidence_grading,
        rag_intent_modes,
        rag_tier_filter,
        retrieval_confidence,
    )

    mode = SimpleNamespace(
        mode="education_rag",
        allowed_tiers=("T1",),
        allowed_use=("patient_education",),
        insufficient_evidence_default="Evidence unavailable.",
    )
    filtered = SimpleNamespace(
        kept_chunks=_valid_evidence_result()["retrieval_context"],
        to_dict=lambda: _valid_evidence_result()["tier_filter"],
    )
    claims = SimpleNamespace(
        to_dict=lambda: _valid_evidence_result()["claim_validation"],
    )
    grade = SimpleNamespace(
        grade="high",
        reasoning="test",
        to_dict=lambda: {"grade": "high", "reasoning": "test"},
    )
    confidence = SimpleNamespace(
        answerability_status="answerable_with_citations",
        evidence_conflict_flag=False,
        reason="test",
        to_dict=lambda: _valid_evidence_result()["retrieval_confidence"],
    )
    monkeypatch.setattr(rag_intent_modes, "select_mode", lambda *_args, **_kwargs: mode)
    monkeypatch.setattr(rag_tier_filter, "filter_chunks_by_mode", lambda *_args, **_kwargs: filtered)
    monkeypatch.setattr(rag_claim_validator, "validate_claims", lambda *_args, **_kwargs: claims)
    monkeypatch.setattr(rag_evidence_grading, "grade_evidence", lambda *_args, **_kwargs: grade)
    monkeypatch.setattr(
        retrieval_confidence,
        "classify_retrieval_uncertainty",
        lambda *_args, **_kwargs: confidence,
    )
    target_module = {
        "rag_intent_modes": rag_intent_modes,
        "rag_tier_filter": rag_tier_filter,
        "rag_claim_validator": rag_claim_validator,
        "rag_evidence_grading": rag_evidence_grading,
        "retrieval_confidence": retrieval_confidence,
    }[module_name]

    def explode(*_args, **_kwargs):
        raise RuntimeError("sensitive fault detail")

    monkeypatch.setattr(target_module, attribute, explode)
    result = _valid_evidence_result()
    original = result["reply"]
    decision = SimpleNamespace(decision="allowed")
    apply_intent_aware_rag_layer(result, result["retrieval_context"], {"status": "passed"}, decision)
    assert result["reply"] != original
    assert result["citations"] == []
    assert result["rag_governance_error"]["stage"] == expected_stage
    assert "sensitive fault detail" not in json.dumps(result)


def test_streaming_never_emits_unenveloped_candidate(monkeypatch):
    # The legacy compatibility router owns helpers imported by the split
    # patient-interactions router, so load it first just as the application does.
    from backend.api.routers import patient as _patient  # noqa: F401
    from backend.api.routers import patient_interactions

    monkeypatch.setattr(
        patient_interactions,
        "handle_patient_chat",
        lambda *_args, **_kwargs: {
            "reply": "UNVALIDATED_STREAM_CANDIDATE",
            "citations": [{"id": "x"}],
            "saved_actions": [],
        },
    )
    events = "".join(patient_interactions._stream_agent_pipeline(
        None,
        "P001",
        "Explain this",
        persist_support_chat=True,
    ))
    answer_deltas = "\n".join(
        line for line in events.splitlines() if line.startswith("data:")
    )
    assert "UNVALIDATED_STREAM_CANDIDATE" not in answer_deltas
    assert EvidenceDisposition.INTERNAL_ERROR.value in events


def test_parse_round_trip_preserves_closed_disposition_enum():
    envelope, error = parse_evidence_envelope(_authorize_valid_result()["evidence_envelope"])
    assert error is None
    assert envelope is not None
    assert envelope.final_disposition is EvidenceDisposition.ALLOW
    assert {member.value for member in EvidenceDisposition} == {
        "ALLOW",
        "ABSTAIN_INSUFFICIENT_EVIDENCE",
        "ABSTAIN_VALIDATION_FAILURE",
        "ABSTAIN_CONFLICTING_EVIDENCE",
        "ABSTAIN_UNSUPPORTED_CLAIMS",
        "BLOCK_MEDICAL_BOUNDARY",
        "BLOCK_SAFETY",
        "INTERNAL_ERROR",
    }
