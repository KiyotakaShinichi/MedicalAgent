"""Immutable contracts for the RAG evidence release boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


EVIDENCE_ENVELOPE_VERSION = "rag_evidence_envelope_v1"
EVIDENCE_POLICY_VERSION = "rag_release_policy_v1"
SAFETY_POLICY_VERSION = "medical_safety_policy_v1"
VALIDATOR_POLICY_VERSION = "claim_citation_validator_v1"


class EvidenceDisposition(str, Enum):
    """Closed release-decision vocabulary.

    Only ``ALLOW`` can authorize an evidence-dependent answer. Other values
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
