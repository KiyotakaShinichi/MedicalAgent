"""Emit all four governance/credibility artifacts.

Writes:

  Data/evals/governance/latest_negative_results_gallery.json
  Data/evals/governance/latest_portfolio_claim_safety_check.json
  Data/evals/governance/latest_eval_contamination_harmonization.json
  Data/evals/models/latest_noisier_synthetic_v2_readiness.json

No retrieval, ML, or live-agent behaviour is changed.  Every artifact
is informational and carries ``clinical_validation: false``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.governance_credibility_artifacts import (  # noqa: E402
    write_eval_contamination_harmonization,
    write_negative_results_gallery,
    write_noisier_synthetic_v2_readiness,
    write_portfolio_claim_safety_check,
)


def main() -> int:
    paths = [
        write_negative_results_gallery(),
        write_portfolio_claim_safety_check(),
        write_eval_contamination_harmonization(),
        write_noisier_synthetic_v2_readiness(),
    ]
    for p in paths:
        print(f"wrote: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
