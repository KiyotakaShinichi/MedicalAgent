# Patient-level temporal cross-validation

NLCare reports model metrics under a CV protocol that is defensible
against two common forms of leakage:

1. **Within-patient leakage** — the same patient's cycle-1 row training
   a model that is then evaluated on the same patient's cycle-6 row.
2. **Future leakage** — a fold that trains on patients whose treatment
   started later than the test patients.

The reference implementation lives in
[`backend/services/patient_temporal_cv.py`](../backend/services/patient_temporal_cv.py)
and is driven by
[`scripts/run_patient_temporal_cv.py`](../scripts/run_patient_temporal_cv.py).
The script writes
[`Data/evals/models/latest_patient_temporal_cv.json`](../Data/evals/models/latest_patient_temporal_cv.json).

## Protocol

- **Group** by `patient_id`. No patient_id appears in both train and
  test in any fold.
- **Order** patients by their earliest `treatment_date`.
- **Walk-forward** across `n_folds` time blocks. Fold `k` uses the k-th
  block as test and all earlier blocks as train (so block 0 is never a
  test set; we get `n_folds - 1` folds total).
- **Censor** train rows dated on or after the held-out fold's first test
  date. This preserves strict row-level chronology in addition to
  patient-level grouping.
- Default model: `GradientBoostingClassifier` over the same feature
  union the production training script uses
  (`NUMERIC_FEATURES + CATEGORICAL_FEATURES`).

## What the JSON contains

Each strategy reports:

- per-fold rows: train/test patient and row counts, date ranges,
  ROC-AUC, Brier, positive-rate train and test;
- aggregates: AUC mean and std, Brier mean and std,
  `patient_overlap_pairs` (lock-in), `temporal_violations` (lock-in),
  `train_rows_censored_after_test_start`, and
  `row_temporal_censoring_applied`;

The top-level `headline.auc_optimism_from_naive_cv` is
`naive_AUC_mean - patient_temporal_AUC_mean`. A positive number means
the naive baseline is optimistically biased relative to the protocol
above.

## What this layer is NOT

- Not clinical validation. Synthetic data only.
- Not a substitute for an external holdout from a different source.
- Not a guarantee against every leakage path (e.g., feature
  construction across the patient's whole trajectory can still leak
  future information into a "pre-cycle" feature; that audit lives in
  `Data/data_lineage`).

## How to extend safely

1. Add a new target — pass `--target ...` to the script. The target
   must be a binary column in `temporal_ml_rows.csv` with both
   classes present.
2. Increase `--n-folds` — only useful up to the point where the test
   block has at least ~30 patients; smaller blocks make AUC noisy.
3. Add the new target's JSON to the release gate config so a
   regression surfaces as a Tier-2 warning.
