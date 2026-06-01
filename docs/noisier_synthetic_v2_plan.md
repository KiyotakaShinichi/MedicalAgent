# Noisier synthetic v2 — readiness scaffold

> **Status: `scaffold_only`.** No noisier synthetic v2 data has been
> generated, no model has been retrained, and no live-agent
> behaviour has been changed by this document. This is engineering
> planning infrastructure; it is **not clinical validation, not
> real-world readiness, and not any kind of model promotion.**

The machine-readable artifact is
[`Data/evals/models/latest_noisier_synthetic_v2_readiness.json`](../Data/evals/models/latest_noisier_synthetic_v2_readiness.json).
Its `scaffold_status` field is test-locked to `scaffold_only` or
`planned_not_trained`; a contributor cannot promote it past that
state without an explicit code change and a failing test
acknowledging the promotion.

## Why current synthetic data is too clean

`Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv` has:

- deterministic labels,
- no missingness,
- no measurement noise,
- no date jitter,
- no reporting bias,
- a balanced subgroup distribution.

This saturates every metric in the MLE stack — toxicity AUC ~1.0,
patient-temporal-CV AUC ~0.9996 — which prevents the ML and
statistical-rigor dimensions of the
[10/10-under-constraints roadmap](ten_out_of_ten_under_constraints.md)
from moving. The roadmap caps `ml_mle_engineering` and
`ml_statistical_rigor` at 7.5 and 6.5 respectively for exactly this
reason.

## Planned noise types

Eight noise mechanisms, in priority order:

1. **missingness_noise** — Bernoulli(p=0.1–0.3) per modality per
   cycle, with patient-block structure (some patients are mostly
   complete; others have systematic missingness).
2. **label_noise** — Symmetric noise rate η ∈ {0.05, 0.10, 0.15} on
   binary outcomes.
3. **measurement_noise** — Multiplicative log-normal noise on lab
   values, calibrated to assay coefficient-of-variation bands.
4. **date_jitter** — Uniform ±3 days per event, preserving ordering.
5. **symptom_reporting_noise** — Per-patient over/under-reporting
   bias drawn once per patient.
6. **imaging_report_ambiguity** — Hedge-word injection at rate
   {0.1, 0.2}; impression/body separation preserved.
7. **treatment_delay_randomness** — Per-cycle delay ~ Geometric(p),
   p tuned so median delay = 0 days and p95 ~ 7 days.
8. **subgroup_distribution_shift** — Reweight subgroup priors per
   release using documented prior shifts.

## Blocked clinical claims (test-locked)

The artifact's `blocked_clinical_claims` field lists what v2 must
never claim:

- "this synthetic v2 represents real patients"
- "this synthetic v2 establishes clinical performance"
- "this synthetic v2 is FDA / IRB ready"
- "this synthetic v2 is sufficient for deployment"
- "this synthetic v2 replaces real-data validation"

The test suite enforces these are present in the artifact.

## Expected evals before any promotion

Before any v2 metric is allowed into a release-gate threshold:

1. Leakage audit re-run with patient-level temporal CV under noise.
2. Subgroup metrics re-run under each noise type independently.
3. Calibration + conformal coverage under noise.
4. Shortcut audit re-run; toxicity AUC must drop below saturation
   (>0.98 is a tripwire).
5. Synthetic data quality proxy with v2-specific disclaimer text.
6. Release gate continues to PASS with v2 artifacts at
   `status: informational`.
7. No metric promoted from monitor-only to treatment-influence.

## Why this remains synthetic-only

Noisier synthetic v2 still has:

- no real patient data,
- no clinician-reviewed labels,
- no IRB.

It improves the **measurement surface** by removing saturation. It
does NOT close the gap to real data. The
`real_clinical_readiness` dimension in the roadmap stays at 1.5/10
regardless of what v2 does.

## What this scaffold does NOT do

- Does not generate data.
- Does not change any model.
- Does not modify the release gate thresholds.
- Does not change retrieval, safety, or live-agent behaviour.
- Does not constitute clinical validation in any form.

## Next steps (gated)

If and when v2 data generation is approved, the following ADRs would
need to be written first:

- ADR 0010 — synthetic v2 noise mechanisms, calibration sources, and
  promotion-blocking rules.
- ADR 0011 — v1-to-v2 transition: when v1 is retained as a
  comparison surface vs replaced.

This document and its artifact are the precondition for those ADRs,
not a substitute for them.

## Related

- [10/10-under-constraints roadmap](ten_out_of_ten_under_constraints.md)
- [Synthetic data quality proxy](synthetic_data_quality.md)
- [Patient-level temporal CV](patient_temporal_cv.md)
- [Negative results gallery](negative_results_gallery.md)
