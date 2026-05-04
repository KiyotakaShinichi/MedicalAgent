# AI Breast Cancer Treatment Monitoring System

AI-assisted proof of concept for longitudinal breast cancer treatment monitoring. The system is not a diagnostic tool. It assumes a patient already has doctor-confirmed breast cancer and helps organize treatment journey data over time.

## What It Does

- Tracks CBC/lab trends across treatment cycles.
- Stores treatment schedules, medication logs, symptoms, imaging reports, and MRI file references.
- Builds patient reports with risk flags, temporal timelines, deterministic monitoring snapshots, and patient-friendly explanations.
- Uses BreastDCEDL/I-SPY1 DCE-MRI features to train a pCR response-classification baseline.
- Adds SHAP explanations for model behavior.
- Saves model artifacts, registry metadata, and prediction audit logs.
- Provides separate patient, clinician, and admin/MLE surfaces.
- Provides a patient support chat that can save symptoms, complete CBC values, medication mentions, and answer timeline-monitoring questions.
- Adds clinician-in-the-loop summary review with approve/edit/reject audit logging.
- Adds admin/MLE analytics for model evaluation, drift checks, champion/challenger comparison, audit counts, and clinician feedback.

## Current Architecture

- `backend/api/main.py`: FastAPI routes.
- `backend/models.py`: SQLAlchemy database schema.
- `backend/services/`: dataset handling, synthetic data, auth, uploads, model artifacts, chat agent.
- `backend/processing/`: clinical trend, risk, timeline, report, and LLM summary logic.
- `frontend/index.html`: clinician dashboard.
- `frontend/patient.html`: patient portal.
- `frontend/admin.html`: admin/MLE operations dashboard.
- `Data/`: generated manifests, model outputs, summaries, and local artifacts.
- `Datasets/`: local real datasets, ignored by git.

## Main Datasets

- QIN-BREAST-02: small breast MRI DICOM and clinical metadata dataset, useful for workflow and DICOM indexing.
- BreastDCEDL I-SPY1: DCE-MRI NIfTI volumes and masks with pCR labels, useful for MRI response-classification proof of concept.
- Synthetic temporal journeys: generated longitudinal CBC, medications, symptoms, treatments, and imaging reports for workflow simulation.
- Complete synthetic journeys: generated end-to-end treatment journeys with diagnosis, treatment sessions, per-cycle MRI, CBC timepoints, medications, symptoms, interventions, and final outcome labels.

## ML Task

Binary treatment-response classification:

- Input: DCE-MRI tumor-region features.
- Output: pCR positive vs pCR negative.

Current best baseline:

- Logistic regression.
- 159 eligible feature rows.
- ROC AUC: 0.637.

The small CNN experiment did not beat the classical baseline, which is documented as an honest result.

## Complete Synthetic Dataset

The complete generated dataset lives in:

```text
Data/complete_synthetic_breast_journeys/
```

It currently includes:

- 300 synthetic `COMPV4-BRCA-*` patients
- 1,800 treatment sessions
- 5,400 CBC rows
- 7,028 medication/support rows
- 3,420 symptom rows
- 2,100 synthetic MRI report rows
- 1,990 support intervention rows
- 300 final outcome rows
- 1,800 training-ready temporal ML rows

Important files:

- `temporal_ml_rows.csv`: cycle-level ML table.
- `outcomes.csv`: final synthetic response/cancer-status labels.
- `interventions.csv`: growth-factor support, transfusion, platelet support, infection management, and urgent support events.
- `data_dictionary.json`: table descriptions.

## Complete Synthetic Model Training

Training script:

```text
python train_complete_synthetic_models.py --target treatment_success_binary
```

Models trained:

- logistic regression
- random forest
- extra trees
- gradient boosting
- SVM
- MLP
- temporal 1D CNN over patient treatment-cycle sequences
- temporal GRU over patient treatment-cycle sequences

Main target:

- `treatment_success_binary`

Best current result on patient-level test split:

- Gradient boosting
- ROC AUC: 0.990
- Average precision: 0.993
- Brier score: 0.062
- Sensitivity: 0.977
- Specificity: 0.875
- Accuracy: 0.933
- Balanced accuracy: 0.926

Other response models:

- Logistic regression patient-level ROC AUC: 0.983
- Temporal 1D CNN patient-level ROC AUC: 0.959
- Temporal GRU patient-level ROC AUC: 0.929

Cycle-level monitoring classifiers were also trained for `toxicity_risk_binary` and `support_intervention_needed`. These are simulator-learning tasks because the labels are generated from CBC/symptom/intervention rules.

Synthetic XAI:

- `Data/complete_synthetic_training/synthetic_xai_explanations.json`
- Explains logistic-regression contributions toward or away from `treatment_success_binary`.
- Used by the patient portal and support agent.

Training notes:

- `Data/complete_synthetic_training_notes.md`

## Safety Positioning

This system does not:

- diagnose cancer
- detect cancer
- confirm metastasis
- choose treatment
- replace clinicians

It is a clinical support and engineering proof of concept for organizing and summarizing longitudinal oncology data.

## Local URLs

Use the current running FastAPI port.

- Clinician dashboard: `/clinician`
- Admin/MLE dashboard: `/admin`
- Patient portal: `/patient`
- API docs: `/docs`

Example:

```text
http://127.0.0.1:8017/patient
http://127.0.0.1:8017/clinician
http://127.0.0.1:8017/admin
```

## Environment

Use the root `.env` file:

```env
GROQ_API_KEY=your_key_here
```

The app also checks the legacy nested `MedicalAgent/.env` location, but the project root is the canonical location.

## Portfolio Description

Built an AI-assisted breast cancer treatment-monitoring platform that combines longitudinal patient records, CBC trends, medication and symptom tracking, breast imaging report NLP, MRI-derived response modeling, XAI explanations, deterministic clinical rule flags, timeline intelligence, and LLM-assisted summaries. Implemented FastAPI services, SQLite persistence, patient/clinician/admin demo sessions, local upload logging, synthetic temporal oncology journeys, model artifact registration, prediction audit trails, clinician-in-the-loop summary review, and admin/MLE monitoring for drift, A/B comparison, and feedback analytics. The system is positioned as clinical decision support and workflow intelligence, not diagnosis or treatment recommendation.
