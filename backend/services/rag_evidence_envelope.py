"""Compatibility facade for fail-closed patient-facing RAG authorization.

The implementation lives in focused ``backend.services.rag_evidence``
modules. Existing import paths remain stable, including the DEP-001 transport
classifier monkeypatch surface used by fault-injection tests.

This is an engineering safety control. It does not establish factual
correctness or clinical validation.
"""

from __future__ import annotations

from typing import Any, MutableMapping

from backend.services.dep001d_output_actionability import classify_output_actionability
from backend.services.rag_evidence.assembly import build_evidence_envelope
from backend.services.rag_evidence.authorization import (
    authorize_evidence_release,
    parse_evidence_envelope,
    validate_cached_response,
)
from backend.services.rag_evidence.enforcement import (
    build_fail_closed_error_result,
    enforce_evidence_release,
    enforce_transport_release as _enforce_transport_release,
)
from backend.services.rag_evidence.metrics import (
    record_rag_cache_rejection,
    snapshot_evidence_release_metrics,
)
from backend.services.rag_evidence.policy import (
    high_risk_semantic_validation_required as _high_risk_semantic_validation_required,  # noqa: F401
)
from backend.services.rag_evidence.responses import build_safe_abstention
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


def enforce_transport_release(
    result: MutableMapping[str, Any],
    *,
    query: str = "",
) -> MutableMapping[str, Any]:
    """Recheck a completed JSON/SSE payload immediately before transport."""

    return _enforce_transport_release(
        result,
        query=query,
        actionability_classifier=classify_output_actionability,
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
