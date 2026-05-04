# Complete Synthetic Model Training Notes

Training source:

- `Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv`

Current generated bundle:

- 300 synthetic `COMPV4-BRCA-*` patients
- 1,800 treatment-cycle ML rows
- 5,400 CBC rows
- 2,100 synthetic MRI report rows
- 1,990 intervention rows
- 300 final outcome rows

Split strategy:

- Patient-level train/test split for journey-level treatment response.
- 225 train patients, 75 test patients.
- Cycles from the same patient are not split across train and test.

Features used:

- cycle number, age, stage, molecular subtype, regimen
- pre-cycle, nadir, and recovery CBC values
- MRI tumor size and percent change from baseline
- symptom burden, intervention count, dose delay, and dose reduction flags

Excluded leakage columns:

- `patient_id`
- `treatment_date`
- `latent_response_strength`
- final outcome/status columns
- all generated label columns

## Target 1: `treatment_success_binary`

Meaning:

- `1`: final synthetic status is no evidence of disease or minimal residual disease.
- `0`: final synthetic status indicates active disease, progression, or continued systemic maintenance.

Patient label distribution:

- Positive: 173 patients
- Negative: 127 patients

Models trained:

- logistic regression
- random forest
- extra trees
- gradient boosting
- SVM
- MLP
- temporal 1D CNN
- temporal GRU

Best model:

- Gradient boosting
- Patient-level ROC AUC: 0.990
- Patient-level accuracy: 0.933
- Patient-level balanced accuracy: 0.926

Other useful baselines:

- Logistic regression patient-level ROC AUC: 0.983
- Temporal 1D CNN patient-level ROC AUC: 0.969
- Temporal GRU patient-level ROC AUC: 0.954

## Cycle-Level Monitoring Targets

Additional tabular classifiers were trained separately:

- `toxicity_risk_binary`
- `support_intervention_needed`

These labels are cycle-level monitoring flags, so sequence CNN/GRU models are skipped for these targets. They are useful for practicing clinical-risk classification, but the high scores mostly show that models learned the simulator rules.

## XAI

Generated file:

- `Data/complete_synthetic_training/synthetic_xai_explanations.json`

Method:

- Logistic-regression contribution approximation after preprocessing.
- Positive contribution means the feature pushed the synthetic model toward `treatment_success_binary=1`.
- Negative contribution means the feature pushed the model away from `treatment_success_binary=1`.

This is model-behavior explanation, not clinical causality.

## Interpretation

The models learned patterns in the synthetic simulator. This is useful for practicing ML engineering, temporal feature design, model comparison, artifact saving, XAI, and patient-facing integration. It is not clinical validation and does not prove real-world treatment prediction.
