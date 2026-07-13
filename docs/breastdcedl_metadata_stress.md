# BreastDCEDL Metadata-Only Stress Benchmark

BreastDCEDL metadata-only stress is an external benchmark probe over public imaging-response metadata. The pCR endpoint is not the same as NLCare synthetic response, toxicity, or monitoring heads. This artifact is not clinical validation, does not update live models, and must not be used for diagnosis, prognosis, treatment recommendation, medication decisions, genetic-risk interpretation, tumor-marker interpretation, or patient-facing prediction.

## Scope

- Metadata-only probe over age, baseline tumor size, HR status, HER2 status, and triple-negative context.
- No image-pixel model training in this artifact.
- pCR is treated as an external stress endpoint, not an NLCare clinical target.

## Result

- Status: `strong`
- Rows: `159`
- Stress result status: `computed`
- ROC AUC: `0.6419`
- Brier: `0.2321`
- Balanced accuracy: `0.6576`

## Decision

- Recommendation: `use_as_external_metadata_stress_only`
- Model promotion allowed: `False`
- Live model update allowed: `False`

## Blocked Claims

- clinical validation
- real patient response prediction
- treatment recommendation
- prognosis or survival prediction
- diagnosis
- tumor-marker interpretation
- genetic-risk interpretation
- model promotion to patient-facing route
