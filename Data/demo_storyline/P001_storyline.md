# Demo Storyline: P001

Use this as a repeatable walkthrough for the patient, clinician, and admin surfaces.

## 1. Login and patient-scoped record
- Show: Login as P001 and confirm the dashboard only exposes that patient journey.
- Say: Role isolation and patient scoping are part of the safety story.
- Evidence: `{"patient_id": "P001", "patient_name": "Patient P001", "diagnosis": "Breast cancer - doctor-confirmed"}`

## 2. Longitudinal treatment timeline
- Show: Scroll the timeline from treatment cycles to labs, symptoms, and imaging.
- Say: The central object is a patient timeline, not a generic chatbot transcript.
- Evidence: `{"treatment_count": 6, "lab_count": 14, "symptom_count": 5, "imaging_count": 3}`

## 3. Deterministic safety before LLM
- Show: Point at CBC and symptom risk flags before opening the support agent.
- Say: CBC and symptom thresholds generate review flags before any language model response.
- Evidence: `{"latest_date": "2026-03-29", "wbc": 4.7, "hemoglobin": 11.1, "platelets": 214.0}`

## 4. Support agent routing
- Show: Ask a casual question, a RAG education question, then provide a clear symptom or lab record.
- Say: The agent routes between conversation, grounded education, and explicit data-entry tools.
- Evidence: `{"casual_prompt": "hi, who are you?", "rag_prompt": "what does pCR mean in breast cancer monitoring?", "tool_prompt": "nausea severity 6/10 today"}`

## 5. Clinician review and audit trail
- Show: Open clinician queue, select the patient, approve/edit/reject the AI summary.
- Say: The system surfaces signals; the clinician remains the decision-maker.
- Evidence: `{"queue_reason": "risk flags and monitoring score drive the review queue"}`

## 6. Admin/MLE governance
- Show: Open eval panels: RAG ablation, agent traces, MLE readiness, calibration, errors, latency.
- Say: The portfolio value is the observable AI lifecycle, not just the final answer.
- Evidence: `{"artifact_paths": ["Data/mle_monitoring/latest_mle_readiness.json", "Data/evals/agent_regression/latest_agent_regression.json", "Data/evals/rag_gold/latest_rag_gold_report.json", "Data/evals/narrative/latest_ai_ml_eval_narrative.md"]}`

## Claim Boundary

Synthetic demo only. The platform is non-diagnostic and requires clinician review.
