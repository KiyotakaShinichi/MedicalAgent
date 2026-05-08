# System Proof and Claim Mapping

Status legend: Implemented = present in code, Partial = present but limited scope or not surfaced end to end, Planned = not implemented yet.

## System proof table
| Capability | Status | Evidence file(s) | Notes / limitations |
| --- | --- | --- | --- |
| Deterministic scope/safety checks | Implemented | [backend/services/agent_rag.py](backend/services/agent_rag.py), [backend/services/security_guardrails.py](backend/services/security_guardrails.py) | Deterministic guardrails run before optional LLM adjudication. |
| Intent routing | Implemented | [backend/services/agent_rag.py](backend/services/agent_rag.py), [backend/services/local_llm.py](backend/services/local_llm.py) | LLM adjudication is optional. |
| Query rewrite/decomposition | Implemented | [backend/services/agent_rag.py](backend/services/agent_rag.py) | Uses normalized and expanded query forms. |
| Hybrid lexical + TF-IDF retrieval | Implemented | [backend/services/rag_vector_index.py](backend/services/rag_vector_index.py) | Local hybrid index. |
| Parent-child context expansion | Implemented | [backend/services/agent_rag.py](backend/services/agent_rag.py) | Expands related snippets. |
| Reranking | Implemented | [backend/services/agent_rag.py](backend/services/agent_rag.py) | Safety and source boosts. |
| Contextual compression | Implemented | [backend/services/agent_rag.py](backend/services/agent_rag.py) | Trims context to size limits. |
| Citation checking | Implemented | [backend/services/agent_rag.py](backend/services/agent_rag.py) | Fails unsafe or uncited answers. |
| Exact cache | Implemented | [backend/services/agent_rag.py](backend/services/agent_rag.py), [backend/models.py](backend/models.py) | Low-risk answers only. |
| Semantic cache | Implemented | [backend/services/agent_rag.py](backend/services/agent_rag.py), [backend/models.py](backend/models.py) | Similarity threshold enforced. |
| TTL expiration | Implemented | [backend/services/agent_rag.py](backend/services/agent_rag.py) | TTL enforced by cache freshness check. |
| KB fingerprint invalidation | Implemented | [backend/services/agent_rag.py](backend/services/agent_rag.py), [backend/services/rag_vector_index.py](backend/services/rag_vector_index.py) | Cache invalidates on KB change. |
| Cache blocking for patient-specific or urgent content | Implemented | [backend/services/agent_rag.py](backend/services/agent_rag.py) | Blocks urgent, treatment-decision, and patient-specific cases. |
| Cautious tool selection for symptoms/CBC/medications/imaging | Implemented | [backend/services/support_chat_agent.py](backend/services/support_chat_agent.py), [backend/services/local_llm.py](backend/services/local_llm.py) | Tools only selected when data entry is clear. |
| Prompt injection guardrails | Implemented | [backend/services/security_guardrails.py](backend/services/security_guardrails.py) | Multilingual and encoded patterns included. |
| Privacy boundary guardrails | Implemented | [backend/services/security_guardrails.py](backend/services/security_guardrails.py) | Blocks cross-patient data requests. |
| Unsafe medical request refusal/escalation | Implemented | [backend/services/agent_rag.py](backend/services/agent_rag.py) | Enforces refusal and escalation language. |
| Synthetic longitudinal modeling | Implemented | [backend/services/complete_synthetic_training.py](backend/services/complete_synthetic_training.py), [backend/services/synthetic_journey.py](backend/services/synthetic_journey.py) | Synthetic data only. |
| Toxicity risk modeling | Partial | [backend/services/complete_synthetic_training.py](backend/services/complete_synthetic_training.py) | Training target exists; no dedicated inference surface. |
| Support-intervention flag modeling | Partial | [backend/services/complete_synthetic_training.py](backend/services/complete_synthetic_training.py) | Training target exists; no dedicated inference surface. |
| BreastDCEDL baseline response classifier | Implemented | [backend/services/breastdcedl_baseline.py](backend/services/breastdcedl_baseline.py), [backend/services/model_artifacts.py](backend/services/model_artifacts.py) | Baseline only, not clinically validated. |
| AUROC/PR-AUC evaluation | Implemented | [backend/services/complete_synthetic_training.py](backend/services/complete_synthetic_training.py), [backend/services/admin_analytics.py](backend/services/admin_analytics.py) | BreastDCEDL baseline provides ROC AUC only. |
| Calibration/Brier evaluation | Implemented | [backend/services/admin_analytics.py](backend/services/admin_analytics.py), [backend/services/evaluation_reports.py](backend/services/evaluation_reports.py) | Synthetic pipeline only. |
| Subgroup checks | Implemented | [backend/services/admin_analytics.py](backend/services/admin_analytics.py) | Synthetic pipeline only. |
| Drift/missingness proxies | Implemented | [backend/services/admin_analytics.py](backend/services/admin_analytics.py), [backend/services/mle_readiness.py](backend/services/mle_readiness.py) | Proxies, not clinical drift detection. |
| Versioned evaluation reports | Implemented | [backend/services/evaluation_reports.py](backend/services/evaluation_reports.py) | Versioned report artifacts. |
| Model artifacts | Implemented | [backend/services/complete_synthetic_training.py](backend/services/complete_synthetic_training.py), [backend/services/model_artifacts.py](backend/services/model_artifacts.py) | PoC artifacts only. |
| Model registry metadata | Implemented | [backend/models.py](backend/models.py), [backend/services/model_artifacts.py](backend/services/model_artifacts.py) | Local registry. |
| Promotion/rollback workflow | Implemented | [backend/services/model_artifacts.py](backend/services/model_artifacts.py) | PoC lifecycle practice. |
| Feature-store materialization | Implemented | [backend/services/feature_store.py](backend/services/feature_store.py) | Local manifest only. |
| Data contracts | Implemented | [backend/services/mle_readiness.py](backend/services/mle_readiness.py) | Enforced for synthetic training data. |
| Audit logging | Implemented | [backend/services/app_logging.py](backend/services/app_logging.py), [backend/models.py](backend/models.py) | Includes prediction and RAG logs. |
| Human-in-the-loop clinician review | Implemented | [backend/services/clinician_feedback.py](backend/services/clinician_feedback.py), [backend/api/main.py](backend/api/main.py) | Approve/edit/reject workflow. |

## Claim-to-file mapping
- Safety-first clinical decision-support PoC for monitoring and clinician review: [SYSTEM_CARD.md](SYSTEM_CARD.md), [backend/processing/patient_state.py](backend/processing/patient_state.py), [backend/api/main.py](backend/api/main.py)
- Deterministic-first RAG pipeline with caching and guardrails: [backend/services/agent_rag.py](backend/services/agent_rag.py), [backend/services/rag_vector_index.py](backend/services/rag_vector_index.py)
- Optional LLM adjudication for routing and cache safety: [backend/services/local_llm.py](backend/services/local_llm.py)
- Multimodal monitoring view and timeline summaries: [backend/services/multimodal_fusion.py](backend/services/multimodal_fusion.py), [backend/services/patient_timeline_summary.py](backend/services/patient_timeline_summary.py)
- ML training, evaluation, and readiness gates: [backend/services/complete_synthetic_training.py](backend/services/complete_synthetic_training.py), [backend/services/mle_readiness.py](backend/services/mle_readiness.py), [backend/services/evaluation_reports.py](backend/services/evaluation_reports.py)
- Model registry, promotion, and rollback: [backend/services/model_artifacts.py](backend/services/model_artifacts.py)
- Human-in-the-loop clinician review workflow: [backend/services/clinician_feedback.py](backend/services/clinician_feedback.py), [backend/api/main.py](backend/api/main.py)
- Audit logs and evaluation telemetry: [backend/services/app_logging.py](backend/services/app_logging.py), [backend/services/rag_analytics.py](backend/services/rag_analytics.py), [backend/models.py](backend/models.py)

## Implementation checklist for partial items
- Add dedicated inference surface and UI cards for toxicity risk models.
- Add dedicated inference surface and UI cards for support-intervention flags.
- Add labeled RAG evaluation datasets for groundedness and citation accuracy.
