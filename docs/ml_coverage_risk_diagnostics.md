# ML Coverage/Risk Diagnostics

`Data/evals/models/latest_ml_coverage_risk_diagnostics.json` summarizes
whether the synthetic ML layer abstains when evidence is weak.

It consumes:

- `Data/evals/models/latest_evidence_abstention_eval.json`
- `Data/evals/models/latest_synthetic_prediction_statistical_audit.json`

## What It Checks

- Full-data rows remain mostly covered.
- Required low-evidence scenarios abstain:
  - no imaging
  - CBC pre-only
  - demographics only
  - symptoms only
- Selective-risk curves exist so reviewers can see how stricter confidence
  margins trade coverage for covered-row accuracy.

## Current Intended Reading

If low-evidence scenarios abstain, the system is behaving more logically under
its own safety contract. It is refusing to invent response-pattern confidence
when required modalities are missing.

This is still synthetic-only evidence. It does not prove real-patient
calibration, real-world safety, or clinical usefulness.

## Claim Boundary

This artifact is not clinical validation, not external validation, not
treatment evidence, and not healthcare production readiness. It is an
engineering diagnostic for abstention behavior under simulator-built data.
