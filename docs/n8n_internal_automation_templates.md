# n8n Internal Automation Templates

n8n workflow templates are internal automation scaffolds only. They may notify maintainers, create admin tickets, remind reviewers, or trigger internal eval refreshes. They must not send PHI, issue medical advice, override safety validators, or automate diagnosis, treatment, medication, prognosis, genetics, tumor-marker, or clinical-escalation decisions.

## Templates

- **Release Gate Alert** (`release_gate_alert`)
  - Path: `Data\evals\ops\n8n_workflow_templates\release_gate_alert.json`
  - Allowed: Notify maintainers when release gate passes or fails.
  - Not allowed: Do not expose patient data or clinical conclusions.

- **Stale Artifact Ticket** (`stale_artifact_ticket`)
  - Path: `Data\evals\ops\n8n_workflow_templates\stale_artifact_ticket.json`
  - Allowed: Create an admin task when critical eval artifacts are stale.
  - Not allowed: Do not downgrade safety thresholds or auto-refresh patient-facing outputs.

- **Reviewer Intake Reminder** (`reviewer_intake_reminder`)
  - Path: `Data\evals\ops\n8n_workflow_templates\reviewer_intake_reminder.json`
  - Allowed: Send reminder/checklist for external reviewer packets.
  - Not allowed: Do not imply clinician approval or clinical validation.

- **Eval Refresh Trigger** (`eval_refresh_trigger`)
  - Path: `Data\evals\ops\n8n_workflow_templates\eval_refresh_trigger.json`
  - Allowed: Trigger internal eval refresh jobs for synthetic/non-live artifacts.
  - Not allowed: Do not trigger clinical actions or use raw patient chat payloads.

- **Trace Quality Digest** (`trace_quality_digest`)
  - Path: `Data\evals\ops\n8n_workflow_templates\trace_quality_digest.json`
  - Allowed: Notify maintainers when trace coverage or trace-envelope validation needs attention.
  - Not allowed: Do not send raw prompts, raw responses, private chain-of-thought, or PHI.

- **Pinecone Shadow Report** (`pinecone_shadow_report`)
  - Path: `Data\evals\ops\n8n_workflow_templates\pinecone_shadow_report.json`
  - Allowed: Send managed-vector shadow retrieval metrics for engineering review.
  - Not allowed: Do not promote Pinecone to live retrieval or store patient chat content.

- **External Red-Team Intake** (`external_red_team_intake`)
  - Path: `Data\evals\ops\n8n_workflow_templates\external_red_team_intake.json`
  - Allowed: Send no-read red-team packet links and attestation checklist to a reviewer.
  - Not allowed: Do not imply clinical validation, clinician approval, or completed external review.

- **Dependency Security Alert** (`dependency_security_alert`)
  - Path: `Data\evals\ops\n8n_workflow_templates\dependency_security_alert.json`
  - Allowed: Create an internal ticket for dependency/security-scan findings.
  - Not allowed: Do not label dependency scan pass/fail as HIPAA, SOC 2, or healthcare compliance.

- **Deployment Health Alert** (`deployment_health_alert`)
  - Path: `Data\evals\ops\n8n_workflow_templates\deployment_health_alert.json`
  - Allowed: Notify maintainers about demo service health and stale engineering artifacts.
  - Not allowed: Do not present engineering health as clinical safety or send patient data.

- **High-Priority Conversation Review Alert** (`high_risk_review_alert`)
  - Path: `Data\evals\ops\n8n_workflow_templates\high_risk_review_alert.json`
  - Allowed: Notify an approved internal reviewer channel that a redacted NLCare review item is waiting. Operators may attach an email, SMS, or Viber node after access-control review.
  - Not allowed: Do not send patient identifiers, raw chat text, medical conclusions, or imply that delivery means a clinician saw or acted on the alert.

## Import Instructions

- Import templates manually into n8n; do not commit credentials.
- Replace placeholder webhook URLs and notification channels in n8n UI.
- Use test webhook URLs first.
- Use signed payloads from NLCare when connecting FastAPI to n8n.
- Keep workflows admin-only until compliance/security review exists.

## Blocked Payload Fields

- `patient_name`
- `patient_id`
- `raw_patient_message`
- `full_chat_transcript`
- `medical_record_number`
- `date_of_birth`
- `address`
- `phone`
- `email`
- `raw_prompt`
- `raw_response`
- `raw_trace`
- `private_chain_of_thought`
- `genetic_variant_details_for_patient_advice`
