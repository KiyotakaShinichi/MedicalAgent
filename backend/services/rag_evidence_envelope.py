"""Fail-closed authorization for patient-facing RAG responses.

The evidence envelope is the final programmatic boundary between a generated
candidate and a patient-visible answer.  It is deliberately independent of the
LLM and prompts: unknown, malformed, incomplete, or failed validation states
cannot authorize an evidence-dependent response.

This is an engineering safety control.  It reduces unsupported-answer release;
it does not establish factual correctness or clinical validation.
"""
from __future__ import annotations

import hashlib
import os
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, MutableMapping, Sequence
from uuid import uuid4


EVIDENCE_ENVELOPE_VERSION = "rag_evidence_envelope_v1"
EVIDENCE_POLICY_VERSION = "rag_release_policy_v1"
SAFETY_POLICY_VERSION = "medical_safety_policy_v1"
VALIDATOR_POLICY_VERSION = "claim_citation_validator_v1"


class EvidenceDisposition(str, Enum):
    """Closed release-decision vocabulary.

    Only ``ALLOW`` can authorize an evidence-dependent answer.  Other values
    describe safe, releasable boundary or abstention responses.
    """

    ALLOW = "ALLOW"
    ABSTAIN_INSUFFICIENT_EVIDENCE = "ABSTAIN_INSUFFICIENT_EVIDENCE"
    ABSTAIN_VALIDATION_FAILURE = "ABSTAIN_VALIDATION_FAILURE"
    ABSTAIN_CONFLICTING_EVIDENCE = "ABSTAIN_CONFLICTING_EVIDENCE"
    ABSTAIN_UNSUPPORTED_CLAIMS = "ABSTAIN_UNSUPPORTED_CLAIMS"
    BLOCK_MEDICAL_BOUNDARY = "BLOCK_MEDICAL_BOUNDARY"
    BLOCK_SAFETY = "BLOCK_SAFETY"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass(frozen=True)
class EvidenceEnvelope:
    request_id: str
    version: str
    policy_version: str
    safety_policy_version: str
    validator_version: str
    evidence_required: bool
    response_kind: str
    retrieval_status: str
    answerability_status: str
    evidence_items: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    source_ids: tuple[str, ...] = field(default_factory=tuple)
    source_metadata: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    passage_references: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    claims: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    claim_to_source_mappings: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    citation_validation_status: str = "not_required"
    claim_support_status: str = "not_required"
    evidence_coverage_status: str = "not_required"
    conflict_status: str = "none"
    safety_validation_status: str = "unknown"
    validation_errors: tuple[str, ...] = field(default_factory=tuple)
    validation_warnings: tuple[str, ...] = field(default_factory=tuple)
    abstention_reason: str | None = None
    final_disposition: EvidenceDisposition = EvidenceDisposition.INTERNAL_ERROR
    candidate_response_digest: str = ""
    response_digest: str = ""
    created_at: str = ""
    trace_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "version": self.version,
            "policy_version": self.policy_version,
            "safety_policy_version": self.safety_policy_version,
            "validator_version": self.validator_version,
            "evidence_required": self.evidence_required,
            "response_kind": self.response_kind,
            "retrieval_status": self.retrieval_status,
            "answerability_status": self.answerability_status,
            "evidence_items": [dict(item) for item in self.evidence_items],
            "source_ids": list(self.source_ids),
            "source_metadata": [dict(item) for item in self.source_metadata],
            "passage_references": [dict(item) for item in self.passage_references],
            "claims": [dict(item) for item in self.claims],
            "claim_to_source_mappings": [dict(item) for item in self.claim_to_source_mappings],
            "citation_validation_status": self.citation_validation_status,
            "claim_support_status": self.claim_support_status,
            "evidence_coverage_status": self.evidence_coverage_status,
            "conflict_status": self.conflict_status,
            "safety_validation_status": self.safety_validation_status,
            "validation_errors": list(self.validation_errors),
            "validation_warnings": list(self.validation_warnings),
            "abstention_reason": self.abstention_reason,
            "final_disposition": self.final_disposition.value,
            "candidate_response_digest": self.candidate_response_digest,
            "response_digest": self.response_digest,
            "created_at": self.created_at,
            "trace_metadata": dict(self.trace_metadata),
            "clinical_validation": False,
        }


@dataclass(frozen=True)
class AuthorizationDecision:
    disposition: EvidenceDisposition
    release_evidence_answer: bool
    release_safe_response: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "release_evidence_answer": self.release_evidence_answer,
            "release_safe_response": self.release_safe_response,
            "reason": self.reason,
        }


_METRICS: Counter[str] = Counter()
_METRICS_LOCK = threading.Lock()


def snapshot_evidence_release_metrics() -> dict[str, int]:
    with _METRICS_LOCK:
        return dict(_METRICS)


def record_rag_cache_rejection() -> None:
    """Increment the PHI-free cache rejection counter."""

    _increment("rag_cache_rejected_total")


def _increment(metric: str) -> None:
    with _METRICS_LOCK:
        _METRICS[metric] += 1


def response_digest(reply: Any) -> str:
    return hashlib.sha256(str(reply or "").encode("utf-8")).hexdigest()


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

    Raw query text and raw evidence passages are intentionally excluded.  The
    envelope carries hashes and source/chunk references so it remains useful
    for auditing without becoming a second PHI-bearing trace store.
    """

    if not isinstance(result, Mapping):
        raise TypeError("result_must_be_mapping")
    request_id = str(request_id or _current_request_id() or f"local-{uuid4().hex}")
    input_guardrails = input_guardrails if isinstance(input_guardrails, Mapping) else {}
    pipeline = result.get("pipeline_trace") if isinstance(result.get("pipeline_trace"), Mapping) else {}
    terminal_step = str(pipeline.get("terminal_step") or "unknown")
    intent = str(result.get("intent") or "unknown")
    mode = str(result.get("rag_mode") or "")
    evidence_required = (
        _requires_evidence(result, terminal_step=terminal_step, intent=intent, mode=mode)
        if evidence_required is None
        else bool(evidence_required)
    )

    errors = _dedupe_codes(validation_errors)
    warnings = _dedupe_codes(validation_warnings)
    governance_error = result.get("rag_governance_error")
    if isinstance(governance_error, Mapping):
        errors = _dedupe_codes([*errors, str(governance_error.get("code") or "rag_governance_failure")])

    chunks = _coerce_chunks(result.get("retrieval_context") or retrieved or [])
    tier_filter = result.get("tier_filter") if isinstance(result.get("tier_filter"), Mapping) else {}
    decisions = tier_filter.get("decisions") if isinstance(tier_filter.get("decisions"), list) else []
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
        if not kept_ids or _chunk_id(chunk) in kept_ids
    ]
    evidence_items, source_metadata, passage_references = _evidence_references(
        governed_chunks,
        decision_by_id,
    )
    source_ids = tuple(dict.fromkeys(
        str(item.get("source_id")) for item in source_metadata if item.get("source_id")
    ))

    claim_validation = result.get("claim_validation") if isinstance(result.get("claim_validation"), Mapping) else {}
    claims, mappings = _claim_references(claim_validation)
    if evidence_required and _high_risk_semantic_validation_required(claims):
        errors = _dedupe_codes([*errors, "high_risk_semantic_validation_required"])
    citation_status = str(claim_validation.get("citation_status") or "missing")
    claim_count = _safe_int(claim_validation.get("claim_count"))
    supported_count = _safe_int(claim_validation.get("supported_count"))
    weak_count = _safe_int(claim_validation.get("weakly_supported_count"))
    unsupported_count = _safe_int(claim_validation.get("unsupported_count"))
    claim_support_status = _claim_support_status(
        evidence_required=evidence_required,
        claim_count=claim_count,
        supported_count=supported_count,
        weak_count=weak_count,
        unsupported_count=unsupported_count,
        mode=mode,
    )
    coverage_status = _coverage_status(
        evidence_required=evidence_required,
        citation_status=citation_status,
        claim_support_status=claim_support_status,
    )

    confidence = result.get("retrieval_confidence") if isinstance(result.get("retrieval_confidence"), Mapping) else {}
    answerability = str(confidence.get("answerability_status") or ("not_required" if not evidence_required else "missing"))
    conflict = "unresolved" if bool(confidence.get("evidence_conflict_flag")) else "none"
    retrieval_status = _retrieval_status(
        evidence_required=evidence_required,
        chunks=governed_chunks,
        evidence_items=evidence_items,
        errors=errors,
    )
    safety_status = _safety_validation_status(result, input_guardrails)

    disposition, reason = _derive_disposition(
        result=result,
        evidence_required=evidence_required,
        retrieval_status=retrieval_status,
        answerability_status=answerability,
        citation_status=citation_status,
        claim_support_status=claim_support_status,
        coverage_status=coverage_status,
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
        response_kind=_response_kind(disposition, evidence_required),
        retrieval_status=retrieval_status,
        answerability_status=answerability,
        evidence_items=tuple(evidence_items),
        source_ids=source_ids,
        source_metadata=tuple(source_metadata),
        passage_references=tuple(passage_references),
        claims=tuple(claims),
        claim_to_source_mappings=tuple(mappings),
        citation_validation_status=citation_status if evidence_required else "not_required",
        claim_support_status=claim_support_status,
        evidence_coverage_status=coverage_status,
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


def authorize_evidence_release(envelope: EvidenceEnvelope | Mapping[str, Any] | None) -> AuthorizationDecision:
    """Return a deny-by-default release decision for an envelope."""

    parsed, error = parse_evidence_envelope(envelope)
    if parsed is None:
        return AuthorizationDecision(
            disposition=EvidenceDisposition.ABSTAIN_VALIDATION_FAILURE,
            release_evidence_answer=False,
            release_safe_response=True,
            reason=error or "invalid_evidence_envelope",
        )
    disposition = parsed.final_disposition
    if disposition is not EvidenceDisposition.ALLOW:
        return AuthorizationDecision(
            disposition=disposition,
            release_evidence_answer=False,
            release_safe_response=True,
            reason=parsed.abstention_reason or disposition.value.lower(),
        )
    if parsed.validation_errors:
        return AuthorizationDecision(
            disposition=EvidenceDisposition.ABSTAIN_VALIDATION_FAILURE,
            release_evidence_answer=False,
            release_safe_response=True,
            reason="allow_envelope_contains_validation_errors",
        )
    if parsed.safety_validation_status != "passed":
        return AuthorizationDecision(
            disposition=EvidenceDisposition.ABSTAIN_VALIDATION_FAILURE,
            release_evidence_answer=False,
            release_safe_response=True,
            reason="allow_envelope_safety_not_passed",
        )
    if parsed.evidence_required:
        if parsed.retrieval_status != "succeeded":
            return _deny("allow_envelope_retrieval_not_succeeded")
        if parsed.answerability_status != "answerable_with_citations":
            return _deny("allow_envelope_answerability_not_authorized")
        if not parsed.evidence_items or not parsed.source_ids:
            return _deny("allow_envelope_missing_evidence")
        if parsed.citation_validation_status != "complete":
            return _deny("allow_envelope_citations_incomplete")
        if parsed.claim_support_status != "complete":
            return _deny("allow_envelope_claim_support_incomplete")
        if parsed.evidence_coverage_status != "complete":
            return _deny("allow_envelope_coverage_incomplete")
        if parsed.conflict_status != "none":
            return AuthorizationDecision(
                disposition=EvidenceDisposition.ABSTAIN_CONFLICTING_EVIDENCE,
                release_evidence_answer=False,
                release_safe_response=True,
                reason="allow_envelope_contains_conflict",
            )
    return AuthorizationDecision(
        disposition=EvidenceDisposition.ALLOW,
        release_evidence_answer=parsed.evidence_required,
        release_safe_response=not parsed.evidence_required,
        reason="validated_allow",
    )


def enforce_evidence_release(
    result: MutableMapping[str, Any],
    *,
    query: str = "",
    retrieved: Sequence[Mapping[str, Any]] | None = None,
    input_guardrails: Mapping[str, Any] | None = None,
    request_id: str | None = None,
    validation_errors: Sequence[str] | None = None,
    validation_warnings: Sequence[str] | None = None,
    evidence_required: bool | None = None,
) -> MutableMapping[str, Any]:
    """Create, authorize, and enforce the final evidence envelope in place."""

    boundary_request_id = str(request_id or _current_request_id() or f"local-{uuid4().hex}")
    try:
        _record_simple_event(
            result,
            "evidence_validation_started",
            request_id=boundary_request_id,
            evidence_required=evidence_required,
        )
        envelope = build_evidence_envelope(
            result,
            query=query,
            retrieved=retrieved,
            input_guardrails=input_guardrails,
            request_id=boundary_request_id,
            validation_errors=validation_errors,
            validation_warnings=validation_warnings,
            evidence_required=evidence_required,
        )
        decision = authorize_evidence_release(envelope)
        _record_simple_event(
            result,
            "evidence_validation_completed",
            request_id=boundary_request_id,
            evidence_required=envelope.evidence_required,
            reason=decision.reason,
        )
        if envelope.validation_errors:
            _record_simple_event(
                result,
                "evidence_validator_failure",
                request_id=boundary_request_id,
                evidence_required=envelope.evidence_required,
                reason="validation_component_failure",
            )
    except Exception as exc:  # noqa: BLE001 - final boundary must deny on all failures
        return build_fail_closed_error_result(
            query=query,
            request_id=boundary_request_id,
            error_code=f"evidence_boundary_exception:{type(exc).__name__}",
            result=result,
        )

    if decision.disposition is not EvidenceDisposition.ALLOW:
        result["reply"] = build_safe_abstention(
            decision.disposition,
            query=query,
            input_guardrails=input_guardrails,
            existing_reply=(
                "" if envelope.evidence_required else str(result.get("reply") or "")
            ),
        )
        result["citations"] = []
    envelope_dict = envelope.to_dict()
    envelope_dict["final_disposition"] = decision.disposition.value
    envelope_dict["abstention_reason"] = None if decision.disposition is EvidenceDisposition.ALLOW else decision.reason
    envelope_dict["response_digest"] = response_digest(result.get("reply"))
    result["evidence_envelope"] = envelope_dict
    result["release_authorization"] = decision.to_dict()
    _record_event(result, "evidence_envelope_created", envelope_dict, decision)
    _record_event(
        result,
        "rag_release_allowed" if decision.disposition is EvidenceDisposition.ALLOW else "rag_release_denied",
        envelope_dict,
        decision,
    )
    if decision.disposition is not EvidenceDisposition.ALLOW:
        _record_simple_event(
            result,
            "rag_abstention_reason",
            request_id=boundary_request_id,
            evidence_required=envelope.evidence_required,
            reason=decision.reason,
        )
    _increment("rag_release_allowed_total" if decision.disposition is EvidenceDisposition.ALLOW else "rag_release_denied_total")
    if decision.disposition is not EvidenceDisposition.ALLOW:
        _increment("rag_abstention_total")
    if decision.disposition is EvidenceDisposition.ABSTAIN_VALIDATION_FAILURE:
        _increment("rag_validation_failure_total")
    if decision.disposition is EvidenceDisposition.ABSTAIN_UNSUPPORTED_CLAIMS:
        _increment("rag_unsupported_claim_total")
    return result


def enforce_transport_release(result: MutableMapping[str, Any], *, query: str = "") -> MutableMapping[str, Any]:
    """Recheck a completed JSON/SSE payload immediately before transport.

    This catches alternate entry points, post-authorization reply mutation,
    and legacy cached objects.  It never tries to reconstruct missing evidence.
    """

    container: MutableMapping[str, Any] = result
    nested = result.get("agent_pipeline")
    if isinstance(nested, MutableMapping):
        container = nested
    envelope_raw = container.get("evidence_envelope") or result.get("evidence_envelope")
    envelope, error = parse_evidence_envelope(envelope_raw)
    reply = result.get("reply") if "reply" in result else container.get("reply")
    if envelope is None or response_digest(reply) != envelope.response_digest:
        reason = error or "response_changed_after_authorization"
        _increment("rag_release_denied_total")
        _increment("rag_validation_failure_total")
        _increment("rag_abstention_total")
        failed = build_fail_closed_error_result(
            query=query,
            request_id=(envelope.request_id if envelope else None),
            error_code=reason,
            result=container,
        )
        safe_reply = failed["reply"]
        result["reply"] = safe_reply
        result["citations"] = []
        result["evidence_envelope"] = failed["evidence_envelope"]
        result["release_authorization"] = failed["release_authorization"]
        if container is not result:
            container["reply"] = safe_reply
            container["citations"] = []
            container["evidence_envelope"] = failed["evidence_envelope"]
            container["release_authorization"] = failed["release_authorization"]
        return result

    decision = authorize_evidence_release(envelope)
    if decision.disposition is EvidenceDisposition.ALLOW:
        return result
    # A non-ALLOW envelope may carry a deterministic safety block or a
    # standardized abstention.  Its digest was bound after that replacement;
    # citations must still be empty at transport time.
    transported_citations = result.get("citations") or container.get("citations") or []
    if transported_citations:
        failed = build_fail_closed_error_result(
            query=query,
            request_id=envelope.request_id,
            error_code="non_allow_payload_contains_citations",
            result=container,
        )
        result["reply"] = failed["reply"]
        result["citations"] = []
        result["evidence_envelope"] = failed["evidence_envelope"]
        result["release_authorization"] = failed["release_authorization"]
        if container is not result:
            container.update({
                "reply": failed["reply"],
                "citations": [],
                "evidence_envelope": failed["evidence_envelope"],
                "release_authorization": failed["release_authorization"],
            })
    return result


def validate_cached_response(
    response: Mapping[str, Any] | None,
    *,
    policy: Mapping[str, Any] | None = None,
) -> tuple[bool, str]:
    """Require a current, ALLOW envelope before a cached answer is served."""

    if not isinstance(response, Mapping):
        return False, "corrupted_cache_entry"
    envelope, error = parse_evidence_envelope(response.get("evidence_envelope"))
    if envelope is None:
        return False, error or "cached_answer_missing_envelope"
    policy = policy if isinstance(policy, Mapping) else {}
    required_policy = {
        "evidence_envelope_version": EVIDENCE_ENVELOPE_VERSION,
        "evidence_policy_version": EVIDENCE_POLICY_VERSION,
        "safety_policy_version": SAFETY_POLICY_VERSION,
        "validator_version": VALIDATOR_POLICY_VERSION,
    }
    for key, expected in required_policy.items():
        if policy.get(key) != expected:
            return False, f"cache_policy_mismatch:{key}"
    if response_digest(response.get("reply")) != envelope.response_digest:
        return False, "cached_response_digest_mismatch"
    decision = authorize_evidence_release(envelope)
    if decision.disposition is not EvidenceDisposition.ALLOW:
        return False, f"cached_disposition_not_allow:{decision.disposition.value}"
    return True, "cache_envelope_valid"


def parse_evidence_envelope(
    value: EvidenceEnvelope | Mapping[str, Any] | None,
) -> tuple[EvidenceEnvelope | None, str | None]:
    if isinstance(value, EvidenceEnvelope):
        return value, None
    if not isinstance(value, Mapping):
        return None, "evidence_envelope_missing_or_malformed"
    required = {
        "request_id", "version", "policy_version", "safety_policy_version",
        "validator_version", "evidence_required", "response_kind",
        "retrieval_status", "answerability_status", "evidence_items",
        "source_ids", "source_metadata", "passage_references", "claims",
        "claim_to_source_mappings", "citation_validation_status",
        "claim_support_status", "evidence_coverage_status", "conflict_status",
        "safety_validation_status", "validation_errors", "validation_warnings",
        "final_disposition", "candidate_response_digest", "response_digest",
        "created_at", "trace_metadata",
    }
    missing = sorted(required - set(value.keys()))
    if missing:
        return None, f"evidence_envelope_missing_fields:{','.join(missing)}"
    if value.get("version") != EVIDENCE_ENVELOPE_VERSION:
        _increment("rag_envelope_version_mismatch_total")
        return None, "evidence_envelope_version_mismatch"
    if value.get("policy_version") != EVIDENCE_POLICY_VERSION:
        return None, "evidence_policy_version_mismatch"
    if value.get("safety_policy_version") != SAFETY_POLICY_VERSION:
        return None, "safety_policy_version_mismatch"
    if value.get("validator_version") != VALIDATOR_POLICY_VERSION:
        return None, "validator_version_mismatch"
    try:
        disposition = EvidenceDisposition(str(value.get("final_disposition")))
    except ValueError:
        return None, "unknown_final_disposition"
    list_fields = (
        "evidence_items", "source_ids", "source_metadata", "passage_references",
        "claims", "claim_to_source_mappings", "validation_errors", "validation_warnings",
    )
    if any(not isinstance(value.get(field_name), (list, tuple)) for field_name in list_fields):
        return None, "evidence_envelope_invalid_collection_field"
    if not isinstance(value.get("trace_metadata"), Mapping):
        return None, "evidence_envelope_invalid_trace_metadata"
    if not isinstance(value.get("evidence_required"), bool):
        return None, "evidence_envelope_invalid_evidence_required"
    if not str(value.get("request_id") or ""):
        return None, "evidence_envelope_missing_request_id"
    return EvidenceEnvelope(
        request_id=str(value["request_id"]),
        version=str(value["version"]),
        policy_version=str(value["policy_version"]),
        safety_policy_version=str(value["safety_policy_version"]),
        validator_version=str(value["validator_version"]),
        evidence_required=value["evidence_required"],
        response_kind=str(value["response_kind"]),
        retrieval_status=str(value["retrieval_status"]),
        answerability_status=str(value["answerability_status"]),
        evidence_items=tuple(dict(item) for item in value["evidence_items"] if isinstance(item, Mapping)),
        source_ids=tuple(str(item) for item in value["source_ids"]),
        source_metadata=tuple(dict(item) for item in value["source_metadata"] if isinstance(item, Mapping)),
        passage_references=tuple(dict(item) for item in value["passage_references"] if isinstance(item, Mapping)),
        claims=tuple(dict(item) for item in value["claims"] if isinstance(item, Mapping)),
        claim_to_source_mappings=tuple(dict(item) for item in value["claim_to_source_mappings"] if isinstance(item, Mapping)),
        citation_validation_status=str(value["citation_validation_status"]),
        claim_support_status=str(value["claim_support_status"]),
        evidence_coverage_status=str(value["evidence_coverage_status"]),
        conflict_status=str(value["conflict_status"]),
        safety_validation_status=str(value["safety_validation_status"]),
        validation_errors=tuple(str(item) for item in value["validation_errors"]),
        validation_warnings=tuple(str(item) for item in value["validation_warnings"]),
        abstention_reason=str(value.get("abstention_reason")) if value.get("abstention_reason") else None,
        final_disposition=disposition,
        candidate_response_digest=str(value["candidate_response_digest"]),
        response_digest=str(value["response_digest"]),
        created_at=str(value["created_at"]),
        trace_metadata=dict(value["trace_metadata"]),
    ), None


def build_safe_abstention(
    disposition: EvidenceDisposition,
    *,
    query: str = "",
    input_guardrails: Mapping[str, Any] | None = None,
    existing_reply: str = "",
) -> str:
    """Return a category-specific English or Taglish safe response."""

    taglish = _is_taglish(query)
    if disposition is EvidenceDisposition.BLOCK_SAFETY:
        # Preserve a deterministic urgent/security response only when an
        # independent input safety gate produced it.
        if existing_reply and _independent_safety_trigger(input_guardrails):
            return existing_reply
        return (
            "Hindi ko maipagpapatuloy ang request na iyon. Matutulungan kita sa sarili mong NLCare records at support questions."
            if taglish else
            "I can't continue with that request. I can help with your own NLCare records and support questions."
        )
    if disposition is EvidenceDisposition.BLOCK_MEDICAL_BOUNDARY:
        return (
            "Hindi ako makakapagbigay ng diagnosis, prognosis, dose, o treatment decision. Dalhin ito sa oncology care team mo para sa patient-specific review."
            if taglish else
            "I can't provide a diagnosis, prognosis, dose, or treatment decision. Please bring this to your oncology care team for patient-specific review."
        )
    if disposition is EvidenceDisposition.ABSTAIN_INSUFFICIENT_EVIDENCE:
        return (
            "Hindi sapat ang na-verify na source support para masagot ko ito nang ligtas. Matutulungan kitang ayusin ang tanong para sa oncology care team mo."
            if taglish else
            "I couldn't find enough verified source support to answer that safely. I can help you organize the question for your oncology care team."
        )
    if disposition is EvidenceDisposition.ABSTAIN_CONFLICTING_EVIDENCE:
        return (
            "Hindi sapat ang pagkakatugma ng available sources para makagawa ako ng ligtas na buod. Paki-review ito sa oncology care team mo."
            if taglish else
            "The available sources are not consistent enough for me to summarize safely. Please review this with your oncology care team."
        )
    if disposition is EvidenceDisposition.ABSTAIN_UNSUPPORTED_CLAIMS:
        return (
            "Hindi na-verify ng sources ang lahat ng kailangang claims, kaya hindi ko ilalabas ang sagot. Maaari nating gawing tanong ito para sa care team mo."
            if taglish else
            "The sources did not verify every required claim, so I won't release the answer. I can help turn this into a question for your care team."
        )
    return (
        "Hindi ko nakumpleto ang evidence checks na kailangan para sa ligtas na sagot ngayon. Subukan ulit mamaya o itanong sa oncology care team mo."
        if taglish else
        "I couldn't complete the evidence checks needed for a safe answer right now. Please try again later or ask your oncology care team."
    )


def build_fail_closed_error_result(
    *,
    query: str = "",
    request_id: str | None = None,
    error_code: str = "internal_pipeline_failure",
    result: MutableMapping[str, Any] | None = None,
) -> MutableMapping[str, Any]:
    """Build a self-contained safe result when the final boundary fails."""

    target = result if isinstance(result, MutableMapping) else {}
    safe_reply = build_safe_abstention(EvidenceDisposition.INTERNAL_ERROR, query=query)
    rid = str(request_id or _current_request_id() or f"local-{uuid4().hex}")
    now = datetime.now(timezone.utc).isoformat()
    digest = response_digest(safe_reply)
    envelope = EvidenceEnvelope(
        request_id=rid,
        version=EVIDENCE_ENVELOPE_VERSION,
        policy_version=EVIDENCE_POLICY_VERSION,
        safety_policy_version=SAFETY_POLICY_VERSION,
        validator_version=VALIDATOR_POLICY_VERSION,
        evidence_required=True,
        response_kind="safe_abstention",
        retrieval_status="failed",
        answerability_status="validation_failure",
        citation_validation_status="failed",
        claim_support_status="failed",
        evidence_coverage_status="failed",
        conflict_status="unknown",
        safety_validation_status="failed",
        validation_errors=(str(error_code)[:160],),
        abstention_reason=str(error_code)[:160],
        final_disposition=EvidenceDisposition.INTERNAL_ERROR,
        candidate_response_digest=response_digest(target.get("reply")),
        response_digest=digest,
        created_at=now,
        trace_metadata={"query_hash": hashlib.sha256(str(query or "").lower().encode("utf-8")).hexdigest()},
    )
    target.update({
        "reply": safe_reply,
        "citations": [],
        "intent": target.get("intent") or "internal_error",
        "safety": target.get("safety") or {"level": "unknown", "scope": "internal_failure", "cache_allowed": False},
        "cache": {"status": "not_stored_validation_failure", "cacheable": False, "reason": str(error_code)[:160]},
        "evidence_envelope": envelope.to_dict(),
        "release_authorization": AuthorizationDecision(
            disposition=EvidenceDisposition.INTERNAL_ERROR,
            release_evidence_answer=False,
            release_safe_response=True,
            reason=str(error_code)[:160],
        ).to_dict(),
    })
    decision = AuthorizationDecision(
        disposition=EvidenceDisposition.INTERNAL_ERROR,
        release_evidence_answer=False,
        release_safe_response=True,
        reason=str(error_code)[:160],
    )
    try:
        _record_event(target, "rag_release_denied", envelope.to_dict(), decision)
    except (TypeError, AttributeError):
        # A malformed observability sink must not defeat the final fail-closed
        # response. Replace it with one PHI-safe event instead.
        target["evidence_envelope_events"] = []
        _record_event(target, "rag_release_denied", envelope.to_dict(), decision)
    _increment("rag_release_denied_total")
    _increment("rag_validation_failure_total")
    _increment("rag_abstention_total")
    return target


def _derive_disposition(
    *,
    result: Mapping[str, Any],
    evidence_required: bool,
    retrieval_status: str,
    answerability_status: str,
    citation_status: str,
    claim_support_status: str,
    coverage_status: str,
    conflict_status: str,
    safety_status: str,
    errors: Sequence[str],
    evidence_items: Sequence[Mapping[str, Any]],
    source_metadata: Sequence[Mapping[str, Any]],
) -> tuple[EvidenceDisposition, str]:
    input_status = str(((result.get("guardrails") or {}).get("input") or {}).get("status") or "")
    safety = result.get("safety") if isinstance(result.get("safety"), Mapping) else {}
    post = result.get("post_gen_validator") if isinstance(result.get("post_gen_validator"), Mapping) else {}
    if input_status == "failed" or str(safety.get("scope") or "") in {"security_boundary", "privacy_boundary"}:
        return EvidenceDisposition.BLOCK_SAFETY, "input_safety_boundary"
    if post.get("decision") == "blocked":
        return EvidenceDisposition.BLOCK_MEDICAL_BOUNDARY, "post_generation_medical_boundary"
    if errors:
        return EvidenceDisposition.ABSTAIN_VALIDATION_FAILURE, "validation_component_failure"
    if safety_status != "passed":
        return EvidenceDisposition.ABSTAIN_VALIDATION_FAILURE, "safety_validation_incomplete"
    if not evidence_required:
        return EvidenceDisposition.ALLOW, "non_evidence_response_validated"
    research = result.get("research_evidence_answerability")
    if isinstance(research, Mapping) and research.get("requires_abstention"):
        return EvidenceDisposition.ABSTAIN_INSUFFICIENT_EVIDENCE, "research_evidence_not_claim_supporting"
    if retrieval_status == "insufficient":
        return EvidenceDisposition.ABSTAIN_INSUFFICIENT_EVIDENCE, "no_governance_compatible_evidence"
    if retrieval_status != "succeeded":
        return EvidenceDisposition.ABSTAIN_VALIDATION_FAILURE, "retrieval_status_not_succeeded"
    if conflict_status != "none" or answerability_status == "conflicting_evidence":
        return EvidenceDisposition.ABSTAIN_CONFLICTING_EVIDENCE, "unresolved_evidence_conflict"
    if answerability_status in {"insufficient_evidence", "answerable_with_limited_context", "clinician_review_required"}:
        return EvidenceDisposition.ABSTAIN_INSUFFICIENT_EVIDENCE, f"answerability:{answerability_status}"
    if answerability_status == "refuse_due_to_safety":
        return EvidenceDisposition.BLOCK_SAFETY, "retrieval_router_safety_refusal"
    if answerability_status != "answerable_with_citations":
        return EvidenceDisposition.ABSTAIN_VALIDATION_FAILURE, "answerability_missing_or_unknown"
    if citation_status in {"partial", "unsupported"} or claim_support_status in {"partial", "unsupported"}:
        return EvidenceDisposition.ABSTAIN_UNSUPPORTED_CLAIMS, "claim_or_citation_support_incomplete"
    if citation_status != "complete" or claim_support_status != "complete" or coverage_status != "complete":
        return EvidenceDisposition.ABSTAIN_VALIDATION_FAILURE, "evidence_coverage_missing_or_unknown"
    if not evidence_items or not source_metadata:
        return EvidenceDisposition.ABSTAIN_INSUFFICIENT_EVIDENCE, "empty_evidence_envelope"
    if any(not _source_metadata_complete(item) for item in source_metadata):
        return EvidenceDisposition.ABSTAIN_VALIDATION_FAILURE, "missing_source_metadata"
    return EvidenceDisposition.ALLOW, "all_required_checks_passed"


def _requires_evidence(result: Mapping[str, Any], *, terminal_step: str, intent: str, mode: str) -> bool:
    if mode in {"education_rag", "record_explanation_rag", "clinician_context_rag"}:
        return True
    if intent in {"education", "patient_timeline_monitoring"} and terminal_step in {"generated", "cache_hit"}:
        return True
    envelope = result.get("evidence_envelope")
    if isinstance(envelope, Mapping):
        return bool(envelope.get("evidence_required"))
    return False


def _retrieval_status(*, evidence_required: bool, chunks, evidence_items, errors) -> str:
    if not evidence_required:
        return "not_required"
    if errors:
        return "failed"
    if not chunks or not evidence_items:
        return "insufficient"
    return "succeeded"


def _safety_validation_status(result: Mapping[str, Any], input_guardrails: Mapping[str, Any]) -> str:
    input_status = str(input_guardrails.get("status") or ((result.get("guardrails") or {}).get("input") or {}).get("status") or "unknown")
    if input_status == "failed":
        return "blocked"
    validation = result.get("validation") if isinstance(result.get("validation"), Mapping) else {}
    output = ((result.get("guardrails") or {}).get("output") or {}) if isinstance(result.get("guardrails"), Mapping) else {}
    post = result.get("post_gen_validator") if isinstance(result.get("post_gen_validator"), Mapping) else {}
    escalation = post.get("answer_tier_escalation")
    if isinstance(escalation, Mapping) and escalation.get("available") is False:
        return "failed"
    if post.get("decision") == "blocked":
        return "blocked"
    if post.get("decision") != "allowed":
        return "failed"
    if validation.get("status") != "passed":
        return "failed"
    if output and output.get("status") != "passed":
        return "failed"
    return "passed"


def _evidence_references(chunks, decision_by_id):
    items: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    for chunk in chunks:
        chunk_id = _chunk_id(chunk)
        decision = decision_by_id.get(chunk_id, {})
        source_id = str(decision.get("source_id") or chunk.get("source_id") or chunk.get("parent_id") or "")
        tier = str(decision.get("tier") or chunk.get("source_tier") or chunk.get("tier") or "")
        allowed_use = decision.get("allowed_use") or chunk.get("allowed_use") or []
        if isinstance(allowed_use, str):
            allowed_use = [allowed_use]
        staleness = str(decision.get("staleness_status") or chunk.get("staleness_status") or chunk.get("staleness") or "unknown")
        item = {
            "chunk_id": chunk_id,
            "source_id": source_id,
            "tier": tier,
            "allowed_use": sorted(str(value) for value in allowed_use),
            "staleness_status": staleness,
            "reference": str(chunk.get("source_url") or chunk.get("source_path") or chunk_id),
        }
        items.append(item)
        metadata.append({
            **item,
            "title": str(chunk.get("title") or chunk.get("source_name") or ""),
            "source_name": str(chunk.get("source_name") or ""),
        })
        references.append({
            "chunk_id": chunk_id,
            "source_id": source_id,
            "title": str(chunk.get("title") or ""),
            "reference": item["reference"],
        })
    return items, metadata, references


def _claim_references(claim_validation: Mapping[str, Any]):
    claims: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    for index, verdict in enumerate(claim_validation.get("verdicts") or []):
        if not isinstance(verdict, Mapping) or verdict.get("is_claim") is False:
            continue
        sentence = str(verdict.get("sentence") or "")
        claim_id = f"claim-{index + 1}-{hashlib.sha256(sentence.encode('utf-8')).hexdigest()[:12]}"
        source_ids = [str(value) for value in verdict.get("supporting_chunk_ids") or [] if value]
        claims.append({
            "claim_id": claim_id,
            "claim_hash": hashlib.sha256(sentence.encode("utf-8")).hexdigest(),
            "claim_type": str(verdict.get("claim_type") or "unknown"),
            "status": str(verdict.get("status") or "unknown"),
            "support_score": _safe_float(verdict.get("support_score")),
            "validation_method": str(verdict.get("validation_method") or "unknown"),
        })
        mappings.append({
            "claim_id": claim_id,
            "source_chunk_ids": source_ids,
            "status": str(verdict.get("status") or "unknown"),
        })
    return claims, mappings


def _high_risk_semantic_validation_required(claims: Sequence[Mapping[str, Any]]) -> bool:
    """Fail closed on high-risk factual claims when strict semantic validation is absent."""
    profile = str(os.environ.get("ENVIRONMENT") or os.environ.get("APP_ENV") or "development").lower()
    configured = os.environ.get("NLCARE_HIGH_RISK_SEMANTIC_VALIDATION_REQUIRED")
    enabled = (
        profile in {"staging", "production", "prod"}
        if configured is None
        else str(configured).strip().lower() in {"1", "true", "yes", "on"}
    )
    if not enabled:
        return False
    high_risk_types = {
        "treatment_or_dose",
        "prognosis_or_outcome",
        "genetic",
        "tumor_marker",
    }
    relevant = [claim for claim in claims if str(claim.get("claim_type")) in high_risk_types]
    return bool(relevant) and any(
        claim.get("validation_method") != "nli_entailment"
        for claim in relevant
    )


def _claim_support_status(*, evidence_required, claim_count, supported_count, weak_count, unsupported_count, mode):
    if not evidence_required:
        return "not_required"
    if claim_count == 0:
        return "complete" if mode == "portal_help_rag" else "missing"
    if unsupported_count:
        return "unsupported"
    if weak_count:
        return "partial"
    return "complete" if supported_count == claim_count else "missing"


def _coverage_status(*, evidence_required, citation_status, claim_support_status):
    if not evidence_required:
        return "not_required"
    if citation_status == "complete" and claim_support_status == "complete":
        return "complete"
    if citation_status in {"partial", "unsupported"} or claim_support_status in {"partial", "unsupported"}:
        return "partial"
    return "missing"


def _source_metadata_complete(item: Mapping[str, Any]) -> bool:
    return bool(
        item.get("chunk_id")
        and item.get("source_id")
        and item.get("tier") in {"T1", "T2", "T3", "T4", "T5"}
        and isinstance(item.get("allowed_use"), list)
        and item.get("staleness_status") not in {None, "", "unknown"}
    )


def _response_kind(disposition: EvidenceDisposition, evidence_required: bool) -> str:
    if disposition is EvidenceDisposition.ALLOW:
        return "evidence_answer" if evidence_required else "deterministic_support"
    if disposition in {EvidenceDisposition.BLOCK_MEDICAL_BOUNDARY, EvidenceDisposition.BLOCK_SAFETY}:
        return "safety_block"
    return "safe_abstention"


def _record_event(result, event_name, envelope, decision) -> None:
    events = result.setdefault("evidence_envelope_events", [])
    if not isinstance(events, list):
        raise TypeError("evidence_event_sink_malformed")
    events.append({
        "event": event_name,
        "request_id": envelope.get("request_id"),
        "disposition": decision.disposition.value,
        "reason": decision.reason,
        "evidence_required": bool(envelope.get("evidence_required")),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


def _record_simple_event(
    result: MutableMapping[str, Any],
    event_name: str,
    *,
    request_id: str,
    evidence_required: bool | None,
    reason: str | None = None,
) -> None:
    events = result.setdefault("evidence_envelope_events", [])
    if not isinstance(events, list):
        raise TypeError("evidence_event_sink_malformed")
    event = {
        "event": event_name,
        "request_id": request_id,
        "evidence_required": evidence_required if isinstance(evidence_required, bool) else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if reason:
        event["reason"] = str(reason)[:160]
    events.append(event)


def _current_request_id() -> str | None:
    try:
        from backend.services.request_context import get_request_id
        return get_request_id()
    except Exception:  # noqa: BLE001
        return None


def _independent_safety_trigger(input_guardrails: Mapping[str, Any] | None) -> bool:
    if not isinstance(input_guardrails, Mapping):
        return False
    return input_guardrails.get("status") == "failed" or str(input_guardrails.get("level") or "") in {"high_risk", "blocked"}


def _is_taglish(query: str) -> bool:
    lower = f" {str(query or '').lower()} "
    tokens = (" hindi ", " wala ", " bakit ", " paano ", " pwede ", " ko ", " mo ", " ako ", " kaya ", " ba ", " gamot ", " sagot ")
    return sum(token in lower for token in tokens) >= 2


def _coerce_chunks(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _chunk_id(chunk: Mapping[str, Any]) -> str:
    return str(chunk.get("id") or chunk.get("chunk_id") or "")


def _dedupe_codes(values: Sequence[str] | None) -> list[str]:
    return list(dict.fromkeys(str(value)[:160] for value in (values or []) if value))


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return round(float(value or 0), 4)
    except (TypeError, ValueError):
        return 0.0


def _deny(reason: str) -> AuthorizationDecision:
    return AuthorizationDecision(
        disposition=EvidenceDisposition.ABSTAIN_VALIDATION_FAILURE,
        release_evidence_answer=False,
        release_safe_response=True,
        reason=reason,
    )


__all__ = [
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
