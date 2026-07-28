# Architecture

## Overview
NLCare is a safety-first, non-diagnostic engineering prototype for breast
cancer monitoring and clinician review. It organizes synthetic labs, imaging
summaries, symptoms, treatment-cycle context, and longitudinal trends while
enforcing explicit claim boundaries, auditability, and guardrails. It is not
clinically validated or production healthcare ready.

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
- Managed vector-store contract: [backend/services/managed_vector_store.py](backend/services/managed_vector_store.py)
- Non-patient data pipeline: [backend/services/data_platform_pipeline.py](backend/services/data_platform_pipeline.py)
- Azure reference infrastructure: [infra/azure/main.bicep](infra/azure/main.bicep)
- Safety guardrails: [backend/services/security_guardrails.py](backend/services/security_guardrails.py)
- ML training and registry: [backend/services/complete_synthetic_training.py](backend/services/complete_synthetic_training.py), [backend/services/model_artifacts.py](backend/services/model_artifacts.py)
- Feature store: [backend/services/feature_store.py](backend/services/feature_store.py)
- Human-in-the-loop review: [backend/services/clinician_feedback.py](backend/services/clinician_feedback.py)
- Audit and evaluation logs: [backend/services/app_logging.py](backend/services/app_logging.py), [backend/models.py](backend/models.py), [backend/services/rag_analytics.py](backend/services/rag_analytics.py)

## Non-diagnostic boundary
- Outputs are monitoring signals and clinician-review flags.
- The system does not diagnose or recommend treatment changes.

## Cloud, data, and vector boundary

The local FAISS/BM25 path remains canonical. Azure AI Search and Pinecone are
optional, network-disabled shadow adapters until frozen comparisons justify a
promotion. The local bronze/silver/gold pipeline processes curated non-patient
knowledge assets only and emits contracts, quarantine records, fingerprints,
and lineage. Azure Bicep now compiles locally and includes opt-in private
networking, workload identity/RBAC, cost alerts, and recovery-retention
contracts. It has not been run through an authenticated subscription
`what-if`, deployed, restore-tested, or load-tested.

See [cloud, data, and vector architecture](cloud_data_vector_architecture.md).
