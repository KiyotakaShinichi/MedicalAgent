# Architecture

## Overview
MedicalAgent is a safety-first clinical decision-support proof of concept for breast cancer monitoring and clinician review. It fuses labs, imaging summaries, symptoms, treatment cycles, and longitudinal trends into a unified clinician view while enforcing non-diagnostic boundaries, auditability, and guardrails.

## End-to-end flow
Frontend / Dashboards
-> Timeline and data-entry tools
-> Deterministic scope and safety gate
-> Intent router
-> RAG / ML / tool workflow
-> Validation and guardrails
-> Clinician review
-> Audit logs
-> Evaluation and MLE dashboard

## Core components and evidence
- Frontend portals: [frontend/patient.html](frontend/patient.html), [frontend/index.html](frontend/index.html), [frontend/admin.html](frontend/admin.html)
- API layer and routing: [backend/api/main.py](backend/api/main.py)
- Timeline and risk processing: [backend/processing/timeline.py](backend/processing/timeline.py), [backend/processing/risk_engine.py](backend/processing/risk_engine.py)
- Clinical summaries and clinician-facing signals: [backend/processing/clinical_summary.py](backend/processing/clinical_summary.py), [backend/services/patient_timeline_summary.py](backend/services/patient_timeline_summary.py)
- RAG agent and retrieval: [backend/services/agent_rag.py](backend/services/agent_rag.py), [backend/services/rag_vector_index.py](backend/services/rag_vector_index.py)
- Safety guardrails: [backend/services/security_guardrails.py](backend/services/security_guardrails.py)
- ML training and registry: [backend/services/complete_synthetic_training.py](backend/services/complete_synthetic_training.py), [backend/services/model_artifacts.py](backend/services/model_artifacts.py)
- Feature store: [backend/services/feature_store.py](backend/services/feature_store.py)
- Human-in-the-loop review: [backend/services/clinician_feedback.py](backend/services/clinician_feedback.py)
- Audit and evaluation logs: [backend/services/app_logging.py](backend/services/app_logging.py), [backend/models.py](backend/models.py), [backend/services/rag_analytics.py](backend/services/rag_analytics.py)

## Non-diagnostic boundary
- Outputs are monitoring signals and clinician-review flags.
- The system does not diagnose or recommend treatment changes.
