# Synthetic timeline drift stress scaffold

> **Synthetic-only.** `review_only_boundary` is permanently `true` and
> test-locked. This is **not** clinical deterioration detection,
> **not** real-world drift monitoring, and **not** clinical
> validation.

The runner:

1. Splits `temporal_ml_rows.csv` into a baseline window (earlier
   cycles) and a recent window (later cycles).
2. Computes a lightweight KS-test p-value between the two windows on
   lab columns and symptom columns; flags if any p < 0.05.
3. Applies three engineered shifts to the recent window with a fixed
   seed: 15% downward lab shift, +2 symptom-severity burst, 30%
   per-row missingness spike on labs.
4. Reports detection rates for the engineered shifts + the
   baseline-vs-recent false-shift rate on unmodified data.

## Files

- Module: [`backend/services/synthetic_timeline_drift_stress.py`](../backend/services/synthetic_timeline_drift_stress.py)
- Script: [`scripts/run_synthetic_timeline_drift_stress.py`](../scripts/run_synthetic_timeline_drift_stress.py)
- Artifact: [`Data/evals/models/latest_synthetic_timeline_drift_stress.json`](../Data/evals/models/latest_synthetic_timeline_drift_stress.json)
- Tests: [`tests/test_frontier_engineering_layers.py`](../tests/test_frontier_engineering_layers.py)

## Current honest result

| Metric | Value |
|---|---:|
| distribution_shift_detection_rate | **1.0** |
| **false_shift_rate_on_baseline_synthetic** | **0.6667** |
| lab_trend_shift_detection | true |
| symptom_trend_shift_detection | true |
| missingness_shift_detection | true |

**Honest finding**: all three engineered shifts fire, but the KS
proxy ALSO fires on 2/3 unmodified-baseline comparisons. The
synthetic data has natural cycle-to-cycle variance that the KS test
treats as a shift. **The 0.667 false-shift rate disqualifies this
scaffold from any monitor-on/monitor-off promotion** — the brief
forbids promoting eval-only scaffolds to live behaviour, and the
false-positive floor here is too high.

## Boundary (test-locked)

- `review_only_boundary == true`.
- `clinical_validation == false`.
- `claim_boundary` explicitly says "NOT clinical deterioration
  detection".
- Not gated as a release-gate blocker.

## Related

- [`docs/noisier_synthetic_v2_plan.md`](noisier_synthetic_v2_plan.md)
- [`docs/noisier_synthetic_v2_stress.md`](noisier_synthetic_v2_stress.md)
- [`docs/negative_results_gallery.md`](negative_results_gallery.md)
