# AI Breast Cancer Treatment Monitoring System

AI-assisted proof of concept for longitudinal breast cancer treatment monitoring. The system is not a diagnostic tool. It assumes a patient already has doctor-confirmed breast cancer and helps organize treatment journey data over time.

## What It Does

- Tracks CBC/lab trends across treatment cycles.
- Stores treatment schedules, medication logs, symptoms, imaging reports, and MRI file references.
- Builds patient reports with risk flags, temporal timelines, deterministic monitoring snapshots, and patient-friendly explanations.
- Uses BreastDCEDL/I-SPY1 MRI-derived tabular features for a small pCR response-classification baseline.
- Adds SHAP explanations for model behavior.
- Saves model artifacts, registry metadata, and prediction audit logs.
- Provides separate patient, clinician, and admin/MLE surfaces.
- Provides a patient support chat that can save symptoms, complete CBC values, medication mentions, and answer timeline-monitoring questions.
- Adds clinician-in-the-loop summary review with approve/edit/reject audit logging.
- Adds admin/MLE analytics for model evaluation, calibration, threshold policies, cost-sensitive error analysis, drift checks, subgroup behavior, audit counts, and clinician feedback.
- Adds product-grade failure handling: constrained inputs, invalid-data responses, insufficient-data summaries, model-unavailable states, and clinician-style fallback language.
- Adds production-thinking telemetry: structured app event logs, prediction/error monitoring, confidence distribution, model version promotion, and rollback controls.

## Product Hardening Roadmap Implemented

Week 1, "Make it Real":

- Demo user sessions expose patient, clinician, and admin role context through `/auth/demo-login` and `/auth/whoami`.
- Patient inputs now enforce CBC, symptom, treatment-cycle, chat, and imaging-report constraints before database writes.
- Invalid values return structured `invalid_data` responses instead of failing silently or saving impossible records.
- Patient reports include a `data_availability` section with missing labs, insufficient timeline depth, unavailable model signal, and clinician-style interpretation guidance.

Week 2, "Production Thinking":

- `app_event_logs` records validation errors, patient-input events, model lifecycle changes, and prediction events.
- Admin/MLE analytics show prediction count, operational failure rate, recent errors, event-type counts, and confidence distribution.
- Model registry supports champion promotion and rollback so model v1/v2 lifecycle behavior can be practiced safely.

Week 3, "Stress & Failure Testing":

- Tests now intentionally cover impossible CBC values, extreme-but-plausible lab warnings, invalid symptom severity, missing longitudinal data, monitoring counters, and rollback behavior.
- Threshold, cost-sensitive, false-negative, calibration, subgroup, and decision-impact metrics remain visible in the admin dashboard for failure-mode review.

## Current Architecture

- `backend/api/main.py`: FastAPI routes.
- `backend/models.py`: SQLAlchemy database schema.
- `backend/services/`: dataset handling, synthetic data, auth, uploads, model artifacts, chat agent.
- `backend/processing/`: clinical trend, risk, timeline, report, and LLM summary logic.
- `frontend/index.html`: clinician dashboard.
- `frontend/patient.html`: patient portal.
- `frontend/admin.html`: admin/MLE operations dashboard.
- `scripts/run_training_pipeline.py`: one-command synthetic training, evaluation-report, and registry pipeline.
- `.github/workflows/ci.yml`: CI workflow for backend compilation and tests.
- `Data/`: generated manifests, model outputs, summaries, and local artifacts.
- `Datasets/`: local real datasets, ignored by git.

## Main Datasets

- QIN-BREAST-02: small breast MRI DICOM and clinical metadata dataset, useful for workflow and DICOM indexing.
- BreastDCEDL I-SPY1: DCE-MRI NIfTI volumes and masks with pCR labels, useful for MRI response-classification proof of concept.
- Synthetic temporal journeys: generated longitudinal CBC, medications, symptoms, treatments, and imaging reports for workflow simulation.
- Complete synthetic journeys: generated end-to-end treatment journeys with diagnosis, treatment sessions, per-cycle MRI, CBC timepoints, medications, symptoms, interventions, and final outcome labels.

## Real MRI Baseline Task

Binary treatment-response classification:

- Input: MRI-derived tabular tumor-region features.
- Output: pCR positive vs pCR negative.

Current best baseline:

- Logistic regression.
- 159 eligible feature rows.
- ROC AUC: 0.637.

The small CNN experiment did not beat the classical baseline, which is documented as an honest result.

This baseline is not the main longitudinal treatment-monitoring model and should not be presented as clinical MRI interpretation.

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

Models trained on synthetic longitudinal rows:

- logistic regression
- random forest
- extra trees
- gradient boosting
- SVM
- MLP
- temporal 1D CNN over patient treatment-cycle sequences: Conv1D encoder over cycle-ordered feature sequences with binary cross-entropy objective
- temporal GRU over patient treatment-cycle sequences: GRU sequence encoder over cycle-ordered features with binary cross-entropy objective

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

## Evaluation and MLE Monitoring

The admin/MLE dashboard reports:

- AUROC, AUPRC, sensitivity, specificity, precision, and Brier score.
- Expected calibration error and calibration bins.
- Bootstrap confidence intervals.
- False-negative review cases.
- Decision-curve net benefit.
- Threshold operating points across multiple cutoffs.
- Cost-sensitive threshold policies for safety-first, balanced, and precision-first review workflows.
- Decision-impact simulation categories for clinician-review routing.
- Subgroup performance by stage, subtype, age band, and treatment regimen.
- Drift, data-quality, data-coverage, and clinician-loop metrics.
- Real-vs-synthetic evidence separation so simulator results are not confused with real-dataset baselines.
- MRI-derived feature inventory documenting current tabular imaging features and the planned raw-MRI boundary.

These metrics are project engineering gates only. They do not prove clinical safety or real-world effectiveness.

Versioned evaluation artifacts can be generated into:

```text
Data/model_evaluation_reports/
```

Each run writes `evaluation_report.json`, calibration bins, threshold operating points, cost-sensitive thresholds, false-negative cases, subgroup metrics, decision-impact categories, data coverage, and a manifest.

## Reproducible Pipeline

Run the complete synthetic training/evaluation pipeline:

```text
python scripts/run_training_pipeline.py
```

To regenerate evaluation artifacts and registry metadata from existing trained outputs:

```text
python scripts/run_training_pipeline.py --skip-training
```

The pipeline validates the synthetic temporal ML table, trains models unless skipped, registers the synthetic champion with dataset/model hashes, writes versioned evaluation reports, and keeps the synthetic-data warning explicit.

## Clinician Workflow

The clinician dashboard includes a review queue that prioritizes urgent deterministic risk flags, unreviewed summaries, missing data warnings, `needs_clinician_review` / `watch_closely` statuses, and top review flags from the patient timeline summary.

The review queue routes attention. It does not diagnose or recommend treatment changes.

## Documentation Cards

- `MODEL_CARD.md`: model purpose, input representation, training objective, evaluation, limitations, and safe positioning.
- `DATA_CARD.md`: dataset sources, labels, feature groups, counts, limitations, and safe dataset language.

## Safety Positioning

This system does not:

- diagnose cancer
- detect cancer
- confirm metastasis
- choose treatment
- replace clinicians

It is a clinical-safety-inspired engineering proof of concept for organizing and summarizing longitudinal oncology data.

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

Use `.env.example` as the safe template. Do not commit real API keys or real patient data.

## Portfolio Description

Built an AI-assisted breast cancer treatment-monitoring platform that combines longitudinal patient records, CBC trends, medication and symptom tracking, breast imaging report NLP, planned multimodal integration using MRI-derived features, deterministic clinical rule flags, timeline intelligence, and LLM-assisted summaries. Implemented FastAPI services, SQLite persistence, patient/clinician/admin demo sessions, local upload logging, synthetic temporal oncology journeys, model artifact registration, prediction audit trails, clinician-in-the-loop summary review, and admin/MLE monitoring for calibration, confidence intervals, drift, subgroup behavior, threshold policy, cost-sensitive error analysis, decision-impact simulation, and feedback analytics. The system is positioned as clinician-review support and workflow intelligence, not diagnosis or treatment recommendation.
