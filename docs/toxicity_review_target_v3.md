# Toxicity Review Target V3

`Data/evals/models/latest_toxicity_review_target_v3.json` is a synthetic-only
target-design artifact for the toxicity review-priority head.

It is not a patient-facing toxicity detector. It is not CTCAE grading. It is
not clinical validation.

## Why V3 Exists

Earlier toxicity targets were too close to shortcut-like simulator structure,
especially nadir-CBC and intervention proxies. V3 is a softer review-priority
candidate that combines:

- symptom severity
- symptom count
- symptom persistence across cycles
- intervention and dose-change context
- pre-cycle vulnerability
- limited/capped nadir-CBC contribution
- recovery failure
- small stochastic noise

The goal is not to make a clinically valid toxicity predictor. The goal is to
reduce exact legacy-rule reconstruction and keep shortcut risk measurable.

## Current Intended Reading

The v3 artifact reports:

- model AUROC against the v3 synthetic target
- legacy-rule accuracy and AUROC against v3
- whether the legacy rule still defines the target
- feature-group correlations with the v3 score
- residual shortcut warnings

Even when the legacy rule no longer defines v3, the target remains
simulator-built. Any high metric should be read as engineering behavior inside
synthetic data, not real adverse-event detection.

## Claim Boundary

Toxicity remains `review_hint_only` and `hold_synthetic_only`. V3 does not
authorize treatment advice, diagnosis, CTCAE grade assignment, real adverse
event detection, patient-facing safety claims, or healthcare production use.
