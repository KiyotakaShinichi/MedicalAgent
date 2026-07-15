# ML Logic/Safety Alignment Audit

`Data/evals/models/latest_ml_logic_safety_alignment.json` is a synthetic-only
review artifact that asks whether NLCare's ML layer behaves logically under its
own safety contract.

It is not a model benchmark, not clinical validation, not external validation,
and not evidence of real-patient benefit. It consolidates existing ML/MLE
artifacts into a small set of safety-aligned checks.

## What It Checks

- Nonclinical promotion policy: every model head must remain monitor-only,
  review-hint-only, context-only, or review-routing-only.
- Evidence sufficiency: weak evidence scenarios should abstain or lower
  confidence rather than inventing certainty.
- Calibration and uncertainty: synthetic calibration and conformal interval
  artifacts must exist and stay bounded.
- Patient-level temporal split hygiene: patient overlap should remain zero,
  and temporal-ordering issues must be visible.
- Counterfactual stability: small plausible perturbations should not create
  extreme flips.
- Noisier synthetic stress: engineered noise must not become promotion
  evidence.
- Shortcut-risk boundaries: suspiciously high simulator metrics, especially
  toxicity AUC, must remain framed as review-only warning signs.
- Statistical audit boundary: bootstrap intervals and paired tests must remain
  synthetic-only.
- Coverage/risk diagnostics: low-evidence scenarios must abstain, while
  selective-risk curves remain synthetic-only engineering evidence.
- Toxicity target v3 boundary: the softer review-priority target must reduce
  legacy-rule dominance while staying review-hint-only.

## Current Interpretation

The audit is expected to be strict. A `needs_attention` result is useful when it
points to a real design issue, such as:

- response classification still reporting high coverage when imaging evidence
  is missing despite weaker accuracy;
- toxicity performance looking too perfect because the synthetic generator can
  encode shortcut-like structure;
- temporal CV artifacts separating patients but still exposing date-ordering
  caveats.

Those are not reasons to hide the ML layer. They are reasons to present it as a
serious synthetic MLE exercise with honest boundaries.

## Best Next ML Improvements Under Synthetic-Only Constraints

1. Keep low-evidence response-pattern scenarios abstained and explain missing
   modality reasons beside model cards.
2. Keep temporal CV row-date censoring locked so train rows cannot occur on or
   after a held-out fold start.
3. Keep toxicity as a softer review-priority target and make shortcut warnings
   visible anywhere toxicity metrics appear.
4. Add patient-block bootstrap if future exports include multiple rows per
   patient.
5. Add pre-registered synthetic stress scenarios before retraining on noisier
   data.

## Claim Boundary

This audit improves engineering defensibility only. It does not prove clinical
calibration, treatment utility, real-world safety, clinician approval, or
production healthcare readiness.
