# Background Eval Worker

Background eval worker is an admin-only automation scaffold for redacted engineering jobs. It is not clinical validation, not healthcare production readiness, and cannot execute diagnosis, treatment, medication, prognosis, genetics, tumor-marker, or clinical-escalation actions.

## Dry-Run Status

- Status: `strong`
- Commands executed: `False`
- Accepted jobs: `3`
- Rejected jobs: `1`

## Allowed Job Types

- `create_stale_artifact_ticket`
- `prepare_reviewer_packet_reminder`
- `publish_dependency_security_alert`
- `publish_deployment_health_alert`
- `publish_external_red_team_intake`
- `publish_pinecone_shadow_report`
- `publish_release_gate_alert`
- `publish_trace_quality_digest`
- `refresh_dependency_security_scan`
- `refresh_eval_history`
- `refresh_external_dataset_matrix`
- `refresh_n8n_templates`
- `refresh_ops_health_snapshot`
- `refresh_pinecone_shadow_retrieval`
- `refresh_platform_control_plane`
- `refresh_release_gate_explanation`
- `refresh_runtime_quality_sentinel`
- `refresh_trace_envelope_v2_eval`
- `run_release_gate`

## Blocked Job Types

- `clinical_escalation_without_human_review`
- `diagnosis`
- `dosage_change`
- `genetic_risk_interpretation`
- `message_patient_directly`
- `prognosis`
- `send_phi_to_external_service`
- `treatment_recommendation`
- `tumor_marker_interpretation`

## Blocked Payload Fields

- `address`
- `date_of_birth`
- `email`
- `full_chat_transcript`
- `genetic_variant_details_for_patient_advice`
- `medical_record_number`
- `patient_id`
- `patient_name`
- `phone`
- `private_chain_of_thought`
- `raw_patient_message`
- `raw_prompt`
- `raw_response`

## Automation Upgrade Path

- Use the admin-only /admin/automation API to enqueue redacted jobs.
- Keep local execution behind NLCARE_AUTOMATION_EXECUTION_ENABLED and bounded subprocess timeouts.
- Use HMAC-signed n8n webhooks for internal notifications only.
- Store job outputs as governance artifacts, never as patient-facing medical advice.
