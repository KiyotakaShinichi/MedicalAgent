"""Governance and credibility artifacts, one module per artifact domain.

Each artifact is pure data: a `build_*` function returns the JSON payload and a
sibling `write_*` persists it. No retrieval, ML, safety, or live-agent
behaviour is involved, and every artifact carries `clinical_validation: false`.

The four domains are deliberately separate modules - they share no logic, only
the claim-boundary invariant in `common` - so a change to one artifact cannot
disturb the others.
"""

from __future__ import annotations

from backend.services.governance_artifacts.common import REQUIRED_CLAIM_BOUNDARY_PHRASE
from backend.services.governance_artifacts.negative_results import (
    NEGATIVE_RESULTS_PATH,
    build_negative_results_gallery,
    write_negative_results_gallery,
)
from backend.services.governance_artifacts.portfolio_claim_safety import (
    ALLOWED_PHRASES,
    BANNED_AFFIRMATIVE_PHRASES,
    PORTFOLIO_PATH,
    build_portfolio_claim_safety_check,
    write_portfolio_claim_safety_check,
)
from backend.services.governance_artifacts.contamination_harmonization import (
    CONTAMINATION_PATH,
    build_eval_contamination_harmonization,
    write_eval_contamination_harmonization,
)
from backend.services.governance_artifacts.noisier_v2_readiness import (
    ALLOWED_NOISIER_V2_STATUS,
    NOISIER_V2_PATH,
    build_noisier_synthetic_v2_readiness,
    write_noisier_synthetic_v2_readiness,
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
