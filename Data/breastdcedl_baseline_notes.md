# BreastDCEDL I-SPY1 Baseline Notes

This baseline is an exploratory proof of concept for the breast MRI modeling pipeline.
It is not a clinically validated model.

## Dataset

- Source: BreastDCEDL I-SPY1 subset from Zenodo.
- Local manifest: `Data/breastdcedl_spy1_manifest.csv`
- Metadata patients: 221
- Patients with all three DCE acquisitions and pCR labels used by the baseline: 159
- Positive pCR cases used: 44
- Negative pCR cases used: 115

## Features

The baseline uses simple masked DCE-MRI features:

- tumor voxel count from the provided mask
- masked mean intensity for `acq0`, `acq1`, and `acq2`
- relative early enhancement: `(acq1 - acq0) / acq0`
- relative delayed enhancement: `(acq2 - acq0) / acq0`
- washout: `(acq2 - acq1) / acq1`
- selected enhancement percentiles
- age and baseline longest diameter

## Model

- Model: logistic regression
- Preprocessing: median imputation and standard scaling
- Evaluation: stratified 5-fold cross-validation

## Result

- Accuracy: 0.491
- Balanced accuracy: 0.486
- ROC AUC: 0.508

## Interpretation

This result is near chance. That is acceptable for the current milestone because the goal was to prove that the pipeline can:

1. map metadata to MRI files and masks,
2. load NIfTI DCE-MRI volumes,
3. extract tumor-region features,
4. train and evaluate a baseline pCR model end to end.

The result should not be presented as a useful clinical predictor. The next modeling step should improve the feature set or move to a small CNN only after careful train/validation splitting and leakage checks.
