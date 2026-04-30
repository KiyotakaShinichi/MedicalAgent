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

## Models

- Models: logistic regression and random forest
- Preprocessing: median imputation and standard scaling
- Evaluation: stratified 5-fold cross-validation

## Result

- Logistic regression:
  - Accuracy: 0.616
  - Balanced accuracy: 0.630
  - ROC AUC: 0.637
- Random forest:
  - Accuracy: 0.654
  - Balanced accuracy: 0.543
  - ROC AUC: 0.579
- Best model by ROC AUC: logistic regression

## Interpretation

This is still exploratory, but it is now a usable proof-of-concept baseline. The goal was to prove that the pipeline can:

1. map metadata to MRI files and masks,
2. load NIfTI DCE-MRI volumes,
3. extract tumor-region features,
4. train and evaluate a baseline pCR model end to end.

The result should not be presented as a clinically useful predictor. It can be presented as an end-to-end treatment-response classification PoC.

## Patient-Level Prediction Output

The baseline also exports out-of-fold patient-level probabilities:

- `Data/breastdcedl_spy1_model_predictions.csv`

These values are used by the multimodal monitoring report as the MRI response branch when the patient exists in BreastDCEDL.

## SHAP Explanation Output

The best classical model also has SHAP explanations:

- `Data/breastdcedl_spy1_shap_explanations.json`

For each patient, the file lists:

- features pushing the model toward pCR
- features pushing the model away from pCR
- a short interpretation rule set

SHAP values should be read as model explanations only. They do not prove that a feature caused a better or worse treatment response.

## CNN Status

PyTorch CPU was installed and a small 2D CNN experiment was run on tumor-centered DCE-MRI slices.

Small CNN setup:

- Input: 3-channel 2D slice using `acq0`, `acq1`, and `acq2`
- Slice choice: axial slice with the largest tumor mask area
- Resize: 128 x 128
- Training subset: 120 patients
- Train/validation split: 90 / 30
- Epochs: 4

Small CNN result:

- Validation accuracy: 0.733
- Validation balanced accuracy: 0.500
- Validation ROC AUC: 0.420

Interpretation: the CNN did not outperform the classical baseline. The high raw accuracy is misleading because the validation set is imbalanced; balanced accuracy of 0.500 suggests majority-class behavior. For the current PoC, logistic regression remains the better baseline.
