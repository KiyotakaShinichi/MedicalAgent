# Noisier synthetic v2 stress benchmark

> **Status**: `scaffold_only` per artifact;
> `global_promotion_decision: reject_or_hold` is permanent. No model
> is retrained, no live inference default changes, and the stress
> benchmark does NOT claim realism.

Companion to
[`docs/noisier_synthetic_v2_plan.md`](noisier_synthetic_v2_plan.md).

- Module: [`backend/services/noisier_synthetic_v2_stress.py`](../backend/services/noisier_synthetic_v2_stress.py)
- Script: [`scripts/run_noisier_synthetic_v2_stress.py`](../scripts/run_noisier_synthetic_v2_stress.py)
- Artifact: [`Data/evals/models/latest_noisier_synthetic_v2_stress.json`](../Data/evals/models/latest_noisier_synthetic_v2_stress.json)
- Tests: [`tests/test_noisier_synthetic_v2_stress.py`](../tests/test_noisier_synthetic_v2_stress.py)

## What the runner does

For each of 8 noise types (missingness, label, measurement,
date_jitter, symptom_reporting, imaging_report_ambiguity,
treatment_delay, subgroup_distribution_shift) it:

1. Loads the clean synthetic
   `temporal_ml_rows.csv` (3,600 rows / 600 patients).
2. Applies the noise function with a fixed seed.
3. Scores the noisy frame with deterministic proxies
   (`proxy_accuracy`, `proxy_brier`, `proxy_auroc`,
   `proxy_regression_mae`, `proxy_abstention_rate`,
   `proxy_shortcut_correlation`).
4. Records the metric deltas vs. the clean baseline.
5. Sets `leakage_status` per noise type:
   `leakage_suspect_metric_too_high_under_noise` if accuracy stays
   above 0.97 under label noise, else
   `no_leakage_tripwire_fired`.

The runner does NOT load any trained model from
`Data/complete_synthetic_training/`. The proxies are stand-ins so
the stress benchmark is fast, deterministic, and never accidentally
loads a saturated model into a stress run.

## What the runner does NOT do

- Does not retrain any classifier or regressor.
- Does not change `monitor_only` or any production policy.
- Does not generate a permanent noisier-v2 dataset.
- Does not claim realism, distribution similarity to real cohorts,
  or clinical predictive validity.

`global_promotion_decision` is `reject_or_hold` for every run, by
design.

## Reading the deltas

Positive deltas on `proxy_brier` and `proxy_regression_mae` mean the
metric degraded under noise — expected and good.

A `leakage_suspect_metric_too_high_under_noise` flag is the tripwire
the brief asked for: if a metric stays > 0.97 under label noise, the
synthetic generator's structural leakage is even worse than the
toxicity feature audit already documents.

## What this is NOT

- Not realism.
- Not clinical evidence.
- Not model promotion.
- Not a substitute for real-data validation.

## Related

- [`docs/noisier_synthetic_v2_plan.md`](noisier_synthetic_v2_plan.md)
- [`docs/synthetic_data_quality.md`](synthetic_data_quality.md)
- [`Data/evals/models/latest_toxicity_feature_audit.json`](../Data/evals/models/latest_toxicity_feature_audit.json)
