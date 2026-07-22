# Duke/TCIA tabular external stress test

This bridge uses the official Duke Breast Cancer MRI clinical and extracted-imaging-feature workbooks as a **separate public-data engineering stress test**. It does not train, validate, or promote NLCare's synthetic monitoring heads.

## Controls

- Both downloaded workbooks are SHA-256 locked.
- Raw subject identifiers are replaced with one-way project-scoped hashes before export.
- Treatment, recurrence, survival, and follow-up fields are excluded from model features.
- The response label is kept only as the external benchmark target.
- Clinical-only, MRI-only, and combined feature sets are compared against a prevalence dummy, logistic regression, and gradient boosting over five repeated stratified splits.
- AUROC, average precision, Brier score, calibration error, per-seed ranges, and paired bootstrap deltas are reported.
- No output is available to patient-facing inference or model promotion.

## Interpretation boundary

The cohort's coded pathologic response after neoadjuvant therapy is not the same target as NLCare's synthetic longitudinal monitoring outputs. Results demonstrate data ingestion, leakage controls, baseline comparison, calibration measurement, and failure-aware reporting only. They are not clinical validation, patient-benefit evidence, or production-healthcare evidence.

Official source: [Duke Breast Cancer MRI, The Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/).
