"""Envelope parsing, authorization, and cache revalidation."""

from __future__ import annotations

from typing import Any, Mapping

from backend.services.rag_evidence.metrics import increment
from backend.services.rag_evidence.types import (
    EVIDENCE_ENVELOPE_VERSION,
    EVIDENCE_POLICY_VERSION,
    SAFETY_POLICY_VERSION,
    VALIDATOR_POLICY_VERSION,
    AuthorizationDecision,
    EvidenceDisposition,
    EvidenceEnvelope,
)
from backend.services.rag_evidence.utilities import response_digest


def authorize_evidence_release(
    envelope: EvidenceEnvelope | Mapping[str, Any] | None,
) -> AuthorizationDecision:
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
        increment("rag_envelope_version_mismatch_total")
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
        claim_to_source_mappings=tuple(
            dict(item) for item in value["claim_to_source_mappings"] if isinstance(item, Mapping)
        ),
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


def _deny(reason: str) -> AuthorizationDecision:
    return AuthorizationDecision(
        disposition=EvidenceDisposition.ABSTAIN_VALIDATION_FAILURE,
        release_evidence_answer=False,
        release_safe_response=True,
        reason=reason,
    )
