# Row-Level Prediction Evidence

Status: synthetic-only MLE statistical evidence.

The row-level prediction evidence export converts the existing synthetic test
prediction table into reviewer-friendly rows with:

- patient_id
- split/fold label
- actual classification label
- actual response-score label
- per-model classification probabilities and correctness flags
- per-model regression predictions and absolute errors
- subgroup/context fields such as stage, molecular subtype, regimen, symptoms,
  nadir CBC, and MRI percent change

It then emits:

- a row-level CSV export
- an export manifest
- exact McNemar/binomial paired classification comparisons
- paired bootstrap regression MAE deltas
- calibration bins with Wilson intervals

Run:

```bash
python scripts/run_row_level_prediction_evidence.py
```

Artifacts:

```text
Data/evals/models/latest_row_level_prediction_export.csv
Data/evals/models/latest_row_level_prediction_export_manifest.json
Data/evals/models/latest_paired_model_comparison.json
Data/evals/models/latest_calibration_uncertainty_report.json
```

Boundary: all rows come from synthetic/internal test predictions. These tests
improve MLE discipline and model-comparison transparency, but they do not
establish clinical validity, treatment utility, or real patient calibration.
