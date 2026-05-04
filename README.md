# AI Breast Cancer Treatment Monitoring System

AI-assisted proof of concept for longitudinal breast cancer treatment monitoring. The system is not a diagnostic tool. It assumes a patient already has doctor-confirmed breast cancer and helps organize treatment journey data over time.

## What It Does

- Tracks CBC/lab trends across treatment cycles.
- Stores treatment schedules, medication logs, symptoms, imaging reports, and MRI file references.
- Builds patient reports with risk flags, temporal timelines, and patient-friendly explanations.
- Uses BreastDCEDL/I-SPY1 DCE-MRI features to train a pCR response-classification baseline.
- Adds SHAP explanations for model behavior.
- Saves model artifacts, registry metadata, and prediction audit logs.
- Provides a clinician/admin dashboard and a separate patient portal.
- Provides a patient support chat that can save symptoms, complete CBC values, and medication mentions.

## Current Architecture

- `backend/api/main.py`: FastAPI routes.
- `backend/models.py`: SQLAlchemy database schema.
- `backend/services/`: dataset handling, synthetic data, auth, uploads, model artifacts, chat agent.
- `backend/processing/`: clinical trend, risk, timeline, report, and LLM summary logic.
- `frontend/index.html`: clinician/admin dashboard.
- `frontend/patient.html`: patient portal.
- `Data/`: generated manifests, model outputs, summaries, and local artifacts.
- `Datasets/`: local real datasets, ignored by git.

## Main Datasets

- QIN-BREAST-02: small breast MRI DICOM and clinical metadata dataset, useful for workflow and DICOM indexing.
- BreastDCEDL I-SPY1: DCE-MRI NIfTI volumes and masks with pCR labels, useful for MRI response-classification proof of concept.
- Synthetic temporal journeys: generated longitudinal CBC, medications, symptoms, treatments, and imaging reports for workflow simulation.

## ML Task

Binary treatment-response classification:

- Input: DCE-MRI tumor-region features.
- Output: pCR positive vs pCR negative.

Current best baseline:

- Logistic regression.
- 159 eligible feature rows.
- ROC AUC: 0.637.

The small CNN experiment did not beat the classical baseline, which is documented as an honest result.

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

- Admin dashboard: `/frontend/index.html`
- Patient portal: `/patient`
- API docs: `/docs`

Example:

```text
http://127.0.0.1:8011/patient
```

## Environment

Use the root `.env` file:

```env
GROQ_API_KEY=your_key_here
```

The app also checks the legacy nested `MedicalAgent/.env` location, but the project root is the canonical location.

## Portfolio Description

Built an AI-assisted breast cancer treatment-monitoring platform that combines longitudinal patient records, CBC trends, medication and symptom tracking, breast imaging report NLP, MRI-derived response modeling, SHAP explanations, and LLM-generated clinical summaries. Implemented FastAPI services, SQLite persistence, patient-scoped demo sessions, local upload logging, synthetic temporal oncology journeys, model artifact registration, prediction audit trails, and separate clinician and patient-facing dashboards. The system is positioned as clinical decision support and workflow intelligence, not diagnosis or treatment recommendation.
