# Breast Cancer Monitoring PoC Summary

This project is an AI-assisted breast cancer monitoring and treatment-response proof of concept.

## Main Problem

The system is not trying to detect cancer or diagnose patients. It assumes the patient already has doctor-confirmed breast cancer.

The core problem is:

> Can we organize longitudinal breast cancer patient data and explore whether MRI-derived features help classify treatment response?

The user-facing mental model is:

> A patient journey repository where a patient can store MRI scans, CBC/lab results, medications, symptoms, profile information, and treatment history; the system then summarizes whether available signals look favorable, mixed, or concerning over time.

## System Modules

1. Patient monitoring dashboard
2. CBC trend tracking
3. Treatment cycle tracking
4. Symptom tracking
5. Breast imaging report NLP
6. MRI DICOM/NIfTI dataset handling
7. Synthetic longitudinal demo data
8. BreastDCEDL treatment-response baseline
9. Multimodal fusion assessment
10. SHAP/XAI explanation for MRI response model
11. Patient-scoped portal endpoints and demo sessions
12. Patient file upload logging
13. Groq-assisted patient support chat with deterministic data capture
14. Model registry and prediction audit history

## Real Datasets

### QIN-BREAST-02

- 13 patients
- Breast MRI DICOM folders
- Clinical spreadsheet with stage, receptor status, treatments, scan completion, and response
- Good for DICOM indexing and monitoring demo
- Too small for real ML training

### BreastDCEDL I-SPY1

- 221 metadata rows
- 709 NIfTI image files
- 537 DCE-MRI volumes
- 172 tumor masks
- pCR treatment-response labels
- Better for treatment-response classification PoC

## Current ML Task

Binary classification:

- Input: DCE-MRI tumor-region features
- Output: pCR positive vs pCR negative

pCR means pathologic complete response, a treatment-response outcome after neoadjuvant therapy and surgery.

## Current Baseline

- Eligible patients used: 159
- pCR positive: 44
- pCR negative: 115
- Best model: logistic regression
- ROC AUC: 0.637

## Small CNN Experiment

A small CPU-trained 2D CNN was also tested using one tumor-centered slice from the three DCE acquisitions.

- Patients used: 120
- Validation rows: 30
- Epochs: 4
- Validation ROC AUC: 0.420
- Validation balanced accuracy: 0.500

The CNN did not beat the classical ML baseline. This is useful because it shows the project compares simple ML and DL approaches honestly instead of assuming a neural network is automatically better.

## Multimodal Fusion Layer

The current patient report combines three signal branches:

1. MRI response branch
   - Uses BreastDCEDL model probabilities when available.
   - Falls back to imaging report NLP when no model output exists.

2. Clinical monitoring branch
   - Uses CBC, treatment-cycle, and risk-flag logic.
   - Tracks whether lab values are real, imported, manual, or synthetic.

3. Symptom branch
   - Uses patient-reported symptom count and maximum severity.

The fusion layer outputs:

- overall monitoring status
- treatment monitoring score from 0 to 100
- MRI response signal
- clinical monitoring signal
- symptom signal
- recommended action
- safety note

This is the first version of the user-facing idea: a patient can upload or enter MRI, CBC, symptoms, and profile data, then receive a structured monitoring summary. It is not treatment advice.

The treatment monitoring score is exploratory:

- 70-100: favorable/on-track signal
- 45-69: mixed or watch closely
- 0-44: lower response signal or clinical concerns

The score combines MRI response evidence with penalties for CBC/risk flags and symptom severity. It is a PoC monitoring score, not a validated clinical score.

## XAI / SHAP Layer

The logistic regression MRI response baseline now exports patient-level SHAP explanations:

- `Data/breastdcedl_spy1_shap_explanations.json`

Interpretation:

- Positive SHAP values push the model toward pCR/favorable complete response.
- Negative SHAP values push the model away from pCR and toward non-pCR.
- Larger absolute values mean stronger model influence.
- SHAP explains model behavior, not clinical causality.

The Groq summary prompt is instructed to explain SHAP in this safe framing and avoid calling features clinically good or bad.

## Model Artifact and Audit Layer

The pCR logistic regression baseline is now also treated like a deployable model artifact:

- Final model artifact: `Data/models/breastdcedl_pcr_logreg_v1.joblib`
- Metadata: `Data/models/breastdcedl_pcr_logreg_v1_metadata.json`
- Registry table: `model_registry`
- Prediction audit table: `prediction_audit_logs`

This adds a more production-like lifecycle:

1. Train the final model on extracted BreastDCEDL features.
2. Save the fitted pipeline as a versioned artifact.
3. Register the artifact path, task, features, metrics, and warning metadata.
4. Serve patient-level pCR probability predictions through the API.
5. Store each prediction with its model version, input reference, output JSON, and SHAP explanation.

This does not make the model clinically valid, but it does make the engineering workflow more realistic and auditable.

## Patient Portal and Access Boundary

The patient portal no longer loads the global patient list. It uses a demo patient-scoped bearer session and calls `/me/...` endpoints:

- `/auth/demo-login`
- `/me/patient-report`
- `/me/chat`
- `/me/uploads`

This is not production authentication, but it introduces the correct server-side shape: patient-facing routes are scoped to the authenticated patient context instead of trusting arbitrary patient IDs in the browser.

## Temporal Synthetic Journey Data

The project now has a richer temporal synthetic generator:

- pre-treatment baseline CBC
- pre-cycle CBC before each treatment
- post-cycle CBC nadir
- CBC recovery checks
- scheduled chemotherapy cycles
- supportive medications
- optional endocrine therapy
- symptoms generated around treatment cycles
- baseline, mid-treatment, and follow-up breast MRI reports

This is useful for engineering and ML practice because it creates longitudinal progression data where labs, treatments, symptoms, medications, and imaging are aligned over time. It remains synthetic demo data and must not be treated as clinical evidence.

## Complete Synthetic Training Dataset

A complete synthetic dataset bundle is available at:

- `Data/complete_synthetic_breast_journeys/`

Generated tables:

- `patients.csv`
- `diagnoses.csv`
- `treatment_sessions.csv`
- `labs.csv`
- `medications.csv`
- `symptoms.csv`
- `mri_reports.csv`
- `interventions.csv`
- `outcomes.csv`
- `temporal_ml_rows.csv`

The current generated bundle contains:

- 300 synthetic `COMPV4-BRCA-*` patients
- 1,800 treatment sessions
- 5,400 CBC rows
- 7,028 medication/supportive-medication rows
- 3,420 symptom rows
- 2,100 synthetic MRI report rows
- 1,990 intervention rows
- 300 final outcome rows
- 1,800 training-ready temporal ML rows

The dataset includes synthetic events such as growth-factor support, blood transfusion, platelet support, infection management, treatment delays, dose reductions, and maintenance/surveillance outcomes. The intended ML table is `temporal_ml_rows.csv`, which joins per-cycle labs, MRI response, symptoms, interventions, dose changes, and final outcome labels.

Important: these are simulated labels for engineering practice. They are not real clinical labels and should not be mixed with real datasets as clinical evidence.

## Complete Synthetic Model Training

Models were trained on `Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv` using a patient-level split:

- 225 train patients
- 75 test patients
- 1,350 train cycle rows
- 450 test cycle rows

Models trained:

- logistic regression
- random forest
- extra trees
- gradient boosting
- SVM
- MLP
- temporal 1D CNN
- temporal GRU

Main target:

- `treatment_success_binary`

Best model:

- gradient boosting
- patient-level ROC AUC: 0.990
- patient-level accuracy: 0.933
- patient-level balanced accuracy: 0.926

Other response models:

- logistic regression patient-level ROC AUC: 0.983
- temporal 1D CNN patient-level ROC AUC: 0.969
- temporal GRU patient-level ROC AUC: 0.954

Additional cycle-level monitoring targets were trained separately:

- `toxicity_risk_binary`
- `support_intervention_needed`

Those cycle-level tasks skip CNN/GRU and use tabular models because the label is per treatment cycle, not one final journey outcome. Results are simulator-learning results, not clinical validation.

Synthetic model explanations were generated at `Data/complete_synthetic_training/synthetic_xai_explanations.json`. They explain logistic-regression feature contributions toward or away from the synthetic treatment-response label and are used by the patient portal and chat agent.

## Groq-Assisted Chat

The support chat keeps deterministic extraction for database writes:

- symptoms
- complete CBC values
- medication mentions
- contextual treatment-response questions

Groq is used only to phrase the supportive reply. If the LLM request fails, the app falls back to the deterministic response. The agent now includes latest labs, recent treatment cycles, MRI impressions, final outcome, synthetic model score, and XAI context. Database writes remain deterministic. The LLM is instructed not to diagnose, not to choose treatment, and not to modify medications.

## Honest Limitation

This is a proof of concept, not a clinically validated model. It shows that the pipeline works end to end: dataset loading, MRI/mask mapping, feature extraction, model training, and metric reporting.
