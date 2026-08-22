"""Pure policy decisions for evidence sufficiency and safe release."""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence, cast

from backend.services.rag_evidence.types import EvidenceDisposition


def derive_disposition(
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
    guardrails = cast(Mapping[str, Any], result.get("guardrails") or {})
    input_guardrails = cast(Mapping[str, Any], guardrails.get("input") or {})
    input_status = str(input_guardrails.get("status") or "")
    safety_value = result.get("safety")
    safety: Mapping[str, Any] = safety_value if isinstance(safety_value, Mapping) else {}
    post_value = result.get("post_gen_validator")
    post: Mapping[str, Any] = post_value if isinstance(post_value, Mapping) else {}
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
    if any(not source_metadata_complete(item) for item in source_metadata):
        return EvidenceDisposition.ABSTAIN_VALIDATION_FAILURE, "missing_source_metadata"
    return EvidenceDisposition.ALLOW, "all_required_checks_passed"


def requires_evidence(result: Mapping[str, Any], *, terminal_step: str, intent: str, mode: str) -> bool:
    if mode in {"education_rag", "record_explanation_rag", "clinician_context_rag"}:
        return True
    if intent in {"education", "patient_timeline_monitoring"} and terminal_step in {"generated", "cache_hit"}:
        return True
    envelope = result.get("evidence_envelope")
    if isinstance(envelope, Mapping):
        return bool(envelope.get("evidence_required"))
    return False


def retrieval_status(*, evidence_required: bool, chunks, evidence_items, errors) -> str:
    if not evidence_required:
        return "not_required"
    if errors:
        return "failed"
    if not chunks or not evidence_items:
        return "insufficient"
    return "succeeded"


def safety_validation_status(result: Mapping[str, Any], input_guardrails: Mapping[str, Any]) -> str:
    guardrails = cast(Mapping[str, Any], result.get("guardrails") or {})
    nested_input = cast(Mapping[str, Any], guardrails.get("input") or {})
    input_status = str(
        input_guardrails.get("status")
        or nested_input.get("status")
        or "unknown"
    )
    if input_status == "failed":
        return "blocked"
    validation_value = result.get("validation")
    validation: Mapping[str, Any] = validation_value if isinstance(validation_value, Mapping) else {}
    guardrails_value = result.get("guardrails")
    output = (
        cast(Mapping[str, Any], guardrails_value.get("output") or {})
        if isinstance(guardrails_value, Mapping)
        else {}
    )
    post_value = result.get("post_gen_validator")
    post: Mapping[str, Any] = post_value if isinstance(post_value, Mapping) else {}
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


def high_risk_semantic_validation_required(claims: Sequence[Mapping[str, Any]]) -> bool:
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


def claim_support_status(
    *,
    evidence_required,
    claim_count,
    supported_count,
    weak_count,
    unsupported_count,
    mode,
):
    if not evidence_required:
        return "not_required"
    if claim_count == 0:
        return "complete" if mode == "portal_help_rag" else "missing"
    if unsupported_count:
        return "unsupported"
    if weak_count:
        return "partial"
    return "complete" if supported_count == claim_count else "missing"


def coverage_status(*, evidence_required, citation_status, claim_support_status):
    if not evidence_required:
        return "not_required"
    if citation_status == "complete" and claim_support_status == "complete":
        return "complete"
    if citation_status in {"partial", "unsupported"} or claim_support_status in {"partial", "unsupported"}:
        return "partial"
    return "missing"


def source_metadata_complete(item: Mapping[str, Any]) -> bool:
    return bool(
        item.get("chunk_id")
        and item.get("source_id")
        and item.get("tier") in {"T1", "T2", "T3", "T4", "T5"}
        and isinstance(item.get("allowed_use"), list)
        and item.get("staleness_status") not in {None, "", "unknown"}
    )


def response_kind(disposition: EvidenceDisposition, evidence_required: bool) -> str:
    if disposition is EvidenceDisposition.ALLOW:
        return "evidence_answer" if evidence_required else "deterministic_support"
    if disposition in {EvidenceDisposition.BLOCK_MEDICAL_BOUNDARY, EvidenceDisposition.BLOCK_SAFETY}:
        return "safety_block"
    return "safe_abstention"
