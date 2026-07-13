# Platform Control Plane Architecture

Platform control-plane architecture is an engineering roadmap and contract artifact. It does not establish clinical validation, real patient safety, patient benefit, clinician approval, IRB approval, HIPAA compliance, FHIR interoperability, or production healthcare readiness. It must not be used to claim diagnosis, prognosis, treatment recommendation, medication advice, genetic-risk interpretation, tumor-marker interpretation, or clinical workflow reduction.

## Agent State Machine

- Status: `contract_ready_not_live_refactor`
- States: `11`
- Transitions: `13`

## RAG Control Plane

- intent_specific_rag_mode_selection
- retrieval_backend_selection
- query_rewrite_or_decomposition
- local_faiss_bm25_or_pinecone_shadow_retrieval
- parent_child_context_expansion
- source_tier_allowed_use_filter
- evidence_sufficiency_grading
- conflict_detection
- citation_window_selection
- claim_source_alignment
- post_generation_validation

## Medical Policy Registry

- `diagnosis_boundary`: blocks diagnosis and diagnosis confirmation
- `treatment_change_boundary`: blocks start/stop/switch/delay treatment instructions
- `dosage_boundary`: blocks medication dosing and dose adjustment
- `prognosis_boundary`: blocks survival or recurrence prediction
- `genetic_vus_boundary`: blocks genetic-risk interpretation and VUS-as-positive claims
- `tumor_marker_boundary`: blocks tumor-marker conclusions such as recurrence proof
- `supplement_pharmacist_boundary`: routes supplement safety/replacement questions to pharmacist or clinician review
- `urgent_symptom_escalation`: routes urgent symptoms to appropriate urgent-care language
- `emotional_distress_support`: allows empathic support without clinical authority
- `privacy_cross_patient_boundary`: blocks PII requests and cross-patient exfiltration
- `prompt_injection_boundary`: blocks instruction override and tool-leak attempts

## ML Feature Store Schema

- `demographics_context`: age, synthetic_subtype, stage_context
- `cbc_labs`: pre_wbc, nadir_wbc, hemoglobin, platelets, anc_proxy, missingness_flags
- `symptoms`: symptom_count, max_symptom_severity, persistence_proxy
- `imaging_context`: mri_percent_change, imaging_available, report_trend_context
- `treatment_context`: cycle_index, dose_delay_context, modality_combination_context
- `biomarker_context`: er_status, pr_status, her2_status, ki67_context
- `governance`: lineage_hash, schema_version, generator_seed, split_id

## Eval Ops Registry

- `hard_blocker`: unsafe leakage on critical routes, medical claim-boundary regression, leakage audit failure, patient-facing clinical overclaim
- `warning`: heldout adversarial below target, retrieval improvement not proven, normal RAG p95 above budget, unsupported_context_rate too high, trace coverage needs_attention
- `supporting`: source-tier ablation, synthetic data quality proxy, external dataset readiness map, n8n/Pinecone optional scaffold
- `informational`: unreviewed packets, future access packets, negative-results gallery, not-completed holdouts

## Trace Envelope V2

- `request_id`
- `correlation_id`
- `patient_id_hash`
- `route`
- `intent`
- `safety_decision`
- `policy_decision`
- `retrieval_backend`
- `source_ids`
- `claim_validation`
- `post_generation_decision`
- `cache_status`
- `latency_ms`
- `estimated_cost`
- `final_policy_status`

## Background Eval Worker

Allowed jobs:
- refresh_non_live_eval_artifact
- run_release_gate
- generate_negative_results_gallery
- create_stale_artifact_ticket
- prepare_reviewer_packet_reminder
- run_pinecone_shadow_dry_run

Blocked jobs:
- diagnosis
- treatment_recommendation
- dosage_change
- prognosis
- clinical_escalation_without_human_review
- send_phi_to_external_service

## Implementation Sequence

1. standardize trace envelope v2 across agent routes (risk: medium)
2. implement agent state-machine event emitter around existing branches (risk: medium)
3. generate policy-registry tests from medical boundary registry (risk: low)
4. create eval ops registry endpoint for admin dashboard (risk: low)
5. version synthetic feature-store schema and row lineage manifests (risk: low)
6. add background eval worker with admin-only jobs (risk: medium)
7. run Pinecone shadow comparison only when credentials are explicitly configured (risk: medium)
8. wire n8n workflow import only for redacted admin events (risk: medium)

## Blocked Claims

- clinical validation
- real-world safety guarantee
- patient benefit
- clinician approval
- IRB approval
- HIPAA compliance
- FHIR interoperability
- production healthcare readiness
- diagnostic authority
- treatment recommendation
- prognosis or survival estimate
- genetic-risk interpretation
- tumor-marker interpretation
