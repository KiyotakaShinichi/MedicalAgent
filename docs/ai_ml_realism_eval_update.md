# AI/ML Realism, Evaluation, and Demo Update

Generated artifacts added in this pass:

- `Data/complete_synthetic_breast_journeys_realism_v2/` - BreastDCEDL-calibrated synthetic candidate dataset.
- `Data/mle_monitoring/synthetic_realism_candidate_report.json` - realism audit for the calibrated candidate.
- `Data/complete_synthetic_training_realism_v2/` - separate training/evaluation run for the calibrated candidate.
- `Data/evals/latency/latest_chat_latency_report.json` - support-agent latency observability report.
- `Data/evals/narrative/latest_ai_ml_eval_narrative.md` - plain-English AI/ML metric interpretation report.
- `Data/demo_storyline/P001_storyline.md` - repeatable walkthrough for patient, clinician, and admin surfaces.

## What Improved

The synthetic generator now supports three realism controls:

- `realism_profile="external_calibrated"`: aligns age, baseline tumor size, and subtype mix to the small public BreastDCEDL/I-SPY1 feature extract.
- `toxicity_profile="realistic"`: reduces unrealistically extreme CBC nadirs while preserving safety-flag examples.
- `missingness_mode="ehr_like"`: introduces structured missingness instead of purely random missing fields.

The calibrated candidate improved broad sim-to-real alignment:

- Age KS: strong
- Baseline tumor size KS: strong
- Molecular subtype JS: passed
- Candidate realism alignment score: acceptable

## What Still Needs Honest Framing

The existing champion remains a synthetic-data model. The calibrated candidate is a better simulator/training candidate, not clinical validation.

The right portfolio framing is:

> We identified a synthetic-to-public distribution gap, changed the simulator to reduce it, trained a separate candidate model, and kept the claim boundary explicit.

Do not frame this as proof of real-world oncology performance.

## Recommended Next Step

Promote the calibrated candidate only after comparing it against the current champion on:

- locked synthetic holdout metrics,
- high-noise robustness,
- temporal generalization,
- external BreastDCEDL/I-SPY1 response baseline,
- clinician-review false-negative cost policy.

If the calibrated candidate wins or matches model metrics while improving realism, it should replace the current synthetic champion.
