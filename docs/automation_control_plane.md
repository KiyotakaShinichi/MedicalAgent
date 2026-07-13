# Automation Control Plane

The automation control plane schedules and summarizes redacted engineering work only. It must not send PHI, message patients, issue medical guidance, trigger unreviewed clinical escalation, or weaken safety gates. It is not clinical validation, compliance certification, or healthcare production readiness.

## Current State

- Status: `strong`
- Commands executed while building artifact: `False`
- Webhooks sent while building artifact: `False`
- Event candidates: `7`

## Schedules

- `nightly_core_eval_refresh` (daily): refresh_trace_envelope_v2_eval, refresh_runtime_quality_sentinel, refresh_eval_history, refresh_release_gate_explanation, run_release_gate
- `weekly_integration_shadow_refresh` (weekly): refresh_pinecone_shadow_retrieval, refresh_n8n_templates, refresh_external_dataset_matrix, refresh_platform_control_plane
- `weekly_security_health_refresh` (weekly): refresh_dependency_security_scan, refresh_ops_health_snapshot
- `biweekly_reviewer_reminder` (every_14_days): prepare_reviewer_packet_reminder

## Runtime Contract

- API access is admin-only.
- Jobs default to dry-run.
- Local execution requires `NLCARE_AUTOMATION_EXECUTION_ENABLED=true`.
- n8n dispatch requires a configured URL, signing secret, and explicit enable flag.
- Scheduled installation remains an operator/deployment choice; no host scheduler is installed automatically.
