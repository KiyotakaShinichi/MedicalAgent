"""Four governance/credibility artifacts.

Each artifact is pure-data: a single function builds the JSON payload,
a sibling write_* function persists it to disk.  No retrieval, ML,
safety, or live-agent behaviour is changed.  Every artifact carries
``clinical_validation: false`` and is gated as informational in the
release gate.

Artifacts
~~~~~~~~~
1. ``build_negative_results_gallery`` — explicit catalogue of negative
   / non-promoted findings already documented elsewhere in the repo.
2. ``build_portfolio_claim_safety_check`` — banned-phrase / allowed-
   phrase guardrails for CV, LinkedIn, README, recruiter, and
   senior-engineer wording.
3. ``build_eval_contamination_harmonization`` — maps every eval
   artifact category (used-for-tuning / frozen / external / synthetic
   / live / informational) and assigns allowed claim strength.
4. ``build_noisier_synthetic_v2_readiness`` — readiness scaffold for
   a noisier synthetic-data v2.  Status is ``scaffold_only``; no
   model is retrained, no clinical behaviour changes.

The implementation is one module per artifact under
:mod:`backend.services.governance_artifacts`; they share nothing but the
claim-boundary invariant, so a change to one artifact cannot disturb another.
This module stays the public import surface and re-exports every name it
exported before the split.
"""

from __future__ import annotations

from backend.services.governance_artifacts import (
    ALLOWED_NOISIER_V2_STATUS,
    ALLOWED_PHRASES,
    BANNED_AFFIRMATIVE_PHRASES,
    CONTAMINATION_PATH,
    NEGATIVE_RESULTS_PATH,
    NOISIER_V2_PATH,
    PORTFOLIO_PATH,
    REQUIRED_CLAIM_BOUNDARY_PHRASE,
    build_eval_contamination_harmonization,
    build_negative_results_gallery,
    build_noisier_synthetic_v2_readiness,
    build_portfolio_claim_safety_check,
    write_eval_contamination_harmonization,
    write_negative_results_gallery,
    write_noisier_synthetic_v2_readiness,
    write_portfolio_claim_safety_check,
)

__all__ = [
    "ALLOWED_NOISIER_V2_STATUS",
    "ALLOWED_PHRASES",
    "BANNED_AFFIRMATIVE_PHRASES",
    "CONTAMINATION_PATH",
    "NEGATIVE_RESULTS_PATH",
    "NOISIER_V2_PATH",
    "PORTFOLIO_PATH",
    "REQUIRED_CLAIM_BOUNDARY_PHRASE",
    "build_eval_contamination_harmonization",
    "build_negative_results_gallery",
    "build_noisier_synthetic_v2_readiness",
    "build_portfolio_claim_safety_check",
    "write_eval_contamination_harmonization",
    "write_negative_results_gallery",
    "write_noisier_synthetic_v2_readiness",
    "write_portfolio_claim_safety_check",
]
