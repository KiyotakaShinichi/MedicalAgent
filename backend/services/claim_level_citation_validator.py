"""Claim-level citation validator facade.

New code should prefer this responsibility-named module. It re-exports the
existing validator so behavior remains backward compatible with
``rag_claim_validator``.
"""

from backend.services.rag_claim_validator import (  # noqa: F401
    ClaimValidationResult,
    ClaimVerdict,
    NLI_CONTRADICTION_THRESHOLD,
    NLI_ENTAILMENT_THRESHOLD,
    SUPPORTED_THRESHOLD,
    WEAKLY_SUPPORTED_THRESHOLD,
    validate_claims,
)

__all__ = [
    "ClaimValidationResult",
    "ClaimVerdict",
    "NLI_CONTRADICTION_THRESHOLD",
    "NLI_ENTAILMENT_THRESHOLD",
    "SUPPORTED_THRESHOLD",
    "WEAKLY_SUPPORTED_THRESHOLD",
    "validate_claims",
]
