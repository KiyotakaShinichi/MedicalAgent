# Restricted Dataset Access Packet

This packet prepares future data-access requests. It does not mean access has been granted, data have been received, or clinical validation has been performed.

## Datasets

### AACR GENIE BPC Breast Cancer

- Priority: `highest_future_value`
- Why it matters: real-world biomarker/genomic, treatment, and outcome context that could test treatment-context features
- Fields requested: patient/case identifier, diagnosis and stage, ER/PR/HER2 and molecular subtype, genomic alterations and test type, cancer-directed drug regimen history, HER2-directed therapy history, response/progression endpoints, overall survival or follow-up outcome fields
- Not requested: free-text notes, direct identifiers, treatment recommendation permissions

Analysis plan:

- map permitted fields into canonical ontology
- run leakage checks before modeling
- compare common-feature A/B candidates
- report failure cases and subgroup calibration
- keep all outputs non-diagnostic and non-treatment-recommending

### SEER breast registry

- Priority: `population_distribution_check`
- Why it matters: stage, subtype, surgery, radiation, and coarse treatment distribution checks
- Fields requested: age group, stage, ER/PR/HER2, surgery, radiation, chemotherapy indicator where available, survival/follow-up fields
- Not requested: patient identifiers, clinical notes, full treatment-regimen inference

Analysis plan:

- compare synthetic cohort distributions against registry priors
- do not train NLCare response models from coarse treatment indicators alone

### SEER-Medicare

- Priority: `future_claims_treatment_sequence_check`
- Why it matters: claims can support richer surgery/radiation/chemo/endocrine/HER2-targeted sequence context in older patients
- Fields requested: diagnosis and staging variables, procedure and treatment claims, drug claims for chemotherapy/endocrine/HER2-targeted agents, radiation claims, follow-up/utilization outcomes
- Not requested: direct identifiers, unbounded clinical note extraction

Analysis plan:

- construct coarse treatment sequence features
- evaluate distribution shift by age and treatment modality
- avoid patient-facing recommendations or clinical utility claims

## Safeguards

- use only de-identified or governed research data
- follow dataset-specific terms and data-use agreements
- store data outside git
- run leakage and target-compatibility checks before modeling
- publish only aggregate engineering metrics
- do not expose patient-level records in demos

## Advisor Review Questions

- Are the target labels clinically meaningful for monitor-only use?
- Which outputs should be hidden from patients and shown only to clinicians?
- Which thresholds require clinician/nurse review before demo use?
- Do failure cases suggest unsafe wording or inappropriate model scope?
