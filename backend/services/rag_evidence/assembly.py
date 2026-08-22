"""Construction of audit-safe evidence envelopes from completed RAG outputs."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
from uuid import uuid4

from backend.services.rag_evidence.policy import (
    claim_support_status,
    coverage_status,
    derive_disposition,
    high_risk_semantic_validation_required,
    requires_evidence,
    response_kind,
    retrieval_status,
    safety_validation_status,
)
from backend.services.rag_evidence.references import claim_references, evidence_references
from backend.services.rag_evidence.types import (
    EVIDENCE_ENVELOPE_VERSION,
    EVIDENCE_POLICY_VERSION,
    SAFETY_POLICY_VERSION,
    VALIDATOR_POLICY_VERSION,
    EvidenceDisposition,
    EvidenceEnvelope,
)
from backend.services.rag_evidence.utilities import (
    chunk_id,
    coerce_chunks,
    current_request_id,
    dedupe_codes,
    response_digest,
    safe_int,
)


def build_evidence_envelope(
    result: Mapping[str, Any],
    *,
    query: str = "",
    retrieved: Sequence[Mapping[str, Any]] | None = None,
    input_guardrails: Mapping[str, Any] | None = None,
    request_id: str | None = None,
    validation_errors: Sequence[str] | None = None,
    validation_warnings: Sequence[str] | None = None,
    evidence_required: bool | None = None,
) -> EvidenceEnvelope:
    """Build a strict envelope from completed pipeline outputs.

    Raw query text and raw evidence passages are intentionally excluded. The
    envelope carries hashes and source/chunk references so it remains useful
    for auditing without becoming a second PHI-bearing trace store.
    """

    if not isinstance(result, Mapping):
        raise TypeError("result_must_be_mapping")
    request_id = str(request_id or current_request_id() or f"local-{uuid4().hex}")
    normalized_input_guardrails: Mapping[str, Any] = (
        input_guardrails if isinstance(input_guardrails, Mapping) else {}
    )
    pipeline_value = result.get("pipeline_trace")
    pipeline: Mapping[str, Any] = pipeline_value if isinstance(pipeline_value, Mapping) else {}
    terminal_step = str(pipeline.get("terminal_step") or "unknown")
    intent = str(result.get("intent") or "unknown")
    mode = str(result.get("rag_mode") or "")
    evidence_required = (
        requires_evidence(result, terminal_step=terminal_step, intent=intent, mode=mode)
        if evidence_required is None
        else bool(evidence_required)
    )

    errors = dedupe_codes(validation_errors)
    warnings = dedupe_codes(validation_warnings)
    governance_error = result.get("rag_governance_error")
    if isinstance(governance_error, Mapping):
        errors = dedupe_codes([*errors, str(governance_error.get("code") or "rag_governance_failure")])

    chunks = coerce_chunks(result.get("retrieval_context") or retrieved or [])
    tier_filter_value = result.get("tier_filter")
    tier_filter: Mapping[str, Any] = tier_filter_value if isinstance(tier_filter_value, Mapping) else {}
    decisions_value = tier_filter.get("decisions")
    decisions = decisions_value if isinstance(decisions_value, list) else []
    kept_ids = {
        str(value)
        for value in (tier_filter.get("kept_chunk_ids") or [])
        if value is not None and str(value)
    }
    decision_by_id = {
        str(item.get("chunk_id")): item
        for item in decisions
        if isinstance(item, Mapping) and item.get("decision") == "kept" and item.get("chunk_id")
    }
    governed_chunks = [
        chunk for chunk in chunks
        if not kept_ids or chunk_id(chunk) in kept_ids
    ]
    evidence_items, source_metadata, passage_references = evidence_references(
        governed_chunks,
        decision_by_id,
    )
    source_ids = tuple(dict.fromkeys(
        str(item.get("source_id")) for item in source_metadata if item.get("source_id")
    ))

    claim_validation_value = result.get("claim_validation")
    claim_validation: Mapping[str, Any] = (
        claim_validation_value if isinstance(claim_validation_value, Mapping) else {}
    )
    claims, mappings = claim_references(claim_validation)
    if evidence_required and high_risk_semantic_validation_required(claims):
        errors = dedupe_codes([*errors, "high_risk_semantic_validation_required"])
    citation_status = str(claim_validation.get("citation_status") or "missing")
    claim_count = safe_int(claim_validation.get("claim_count"))
    supported_count = safe_int(claim_validation.get("supported_count"))
    weak_count = safe_int(claim_validation.get("weakly_supported_count"))
    unsupported_count = safe_int(claim_validation.get("unsupported_count"))
    support_status = claim_support_status(
        evidence_required=evidence_required,
        claim_count=claim_count,
        supported_count=supported_count,
        weak_count=weak_count,
        unsupported_count=unsupported_count,
        mode=mode,
    )
    current_coverage_status = coverage_status(
        evidence_required=evidence_required,
        citation_status=citation_status,
        claim_support_status=support_status,
    )

    confidence_value = result.get("retrieval_confidence")
    confidence: Mapping[str, Any] = (
        confidence_value if isinstance(confidence_value, Mapping) else {}
    )
    answerability = str(
        confidence.get("answerability_status")
        or ("not_required" if not evidence_required else "missing")
    )
    conflict = "unresolved" if bool(confidence.get("evidence_conflict_flag")) else "none"
    current_retrieval_status = retrieval_status(
        evidence_required=evidence_required,
        chunks=governed_chunks,
        evidence_items=evidence_items,
        errors=errors,
    )
    safety_status = safety_validation_status(result, normalized_input_guardrails)

    disposition, reason = derive_disposition(
        result=result,
        evidence_required=evidence_required,
        retrieval_status=current_retrieval_status,
        answerability_status=answerability,
        citation_status=citation_status,
        claim_support_status=support_status,
        coverage_status=current_coverage_status,
        conflict_status=conflict,
        safety_status=safety_status,
        errors=errors,
        evidence_items=evidence_items,
        source_metadata=source_metadata,
    )
    candidate_digest = response_digest(result.get("reply"))
    return EvidenceEnvelope(
        request_id=request_id,
        version=EVIDENCE_ENVELOPE_VERSION,
        policy_version=EVIDENCE_POLICY_VERSION,
        safety_policy_version=SAFETY_POLICY_VERSION,
        validator_version=VALIDATOR_POLICY_VERSION,
        evidence_required=evidence_required,
        response_kind=response_kind(disposition, evidence_required),
        retrieval_status=current_retrieval_status,
        answerability_status=answerability,
        evidence_items=tuple(evidence_items),
        source_ids=source_ids,
        source_metadata=tuple(source_metadata),
        passage_references=tuple(passage_references),
        claims=tuple(claims),
        claim_to_source_mappings=tuple(mappings),
        citation_validation_status=citation_status if evidence_required else "not_required",
        claim_support_status=support_status,
        evidence_coverage_status=current_coverage_status,
        conflict_status=conflict,
        safety_validation_status=safety_status,
        validation_errors=tuple(errors),
        validation_warnings=tuple(warnings),
        abstention_reason=None if disposition is EvidenceDisposition.ALLOW else reason,
        final_disposition=disposition,
        candidate_response_digest=candidate_digest,
        response_digest=candidate_digest,
        created_at=datetime.now(timezone.utc).isoformat(),
        trace_metadata={
            "terminal_step": terminal_step,
            "intent": intent,
            "rag_mode": mode or None,
            "query_hash": hashlib.sha256(str(query or "").strip().lower().encode("utf-8")).hexdigest(),
        },
    )
