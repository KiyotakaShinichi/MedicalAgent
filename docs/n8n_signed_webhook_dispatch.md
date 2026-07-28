# Signed n8n Webhook Dispatch

NLCare emits only redacted engineering events to n8n. The sender builds a canonical JSON envelope and signs it with HMAC-SHA256 using `N8N_WEBHOOK_SIGNING_SECRET`. Nested blocked fields are rejected before signing.

Required variables for an explicitly enabled local/test dispatch:

- `N8N_WEBHOOK_DISPATCH_ENABLED=true`
- `N8N_WEBHOOK_BASE_URL=https://<n8n-host>/webhook/nlcare`
- `N8N_WEBHOOK_SIGNING_SECRET=<secret stored outside git>`
- `NLCARE_ALERT_TEST_RECIPIENT_ONLY=true` for the high-risk review template

Plain HTTP is accepted only for localhost development. Imported templates remain inactive. Before activation, configure receiver-side HMAC verification against the raw request body, enforce the `X-NLCare-Timestamp` freshness window, and reject missing, invalid, or replayed event IDs. The generated template checks the redacted payload contract; receiver-side cryptographic verification still requires operator configuration.

Allowed events cover release-gate alerts, stale-artifact tickets, reviewer reminders, eval refresh notifications, trace-quality digests, Pinecone shadow reports, external red-team intake, dependency alerts, demo deployment health, and a redacted high-priority conversation review alert. They cannot carry PHI, raw chat, raw prompts/responses, private chain-of-thought, or clinical instructions.

## High-priority conversation workflow

`high_risk_review_alert` is created only after NLCare has made its own deterministic safety-routing decision. n8n is a notification adapter, never the safety classifier. Its event excludes patient ID, name, contact details, and raw conversation text.

This repository restricts the template to a synthetic test recipient. A successful webhook response means the workflow accepted the event. A signed delivery receipt means only that the configured test channel reported acceptance or delivery. Neither proves that a clinician received, read, assessed, or acted on it.

The local outbox records bounded exponential retries, append-only attempt evidence, dead-letter state, and a signed delivery receipt separately from clinician acknowledgement. The patient-facing response still directs immediate danger to local emergency services rather than asking the patient to wait for a portal notification.

Real email, SMS, or Viber recipients remain outside the prototype until privacy, consent, security, on-call ownership, and clinical-governance requirements are independently satisfied.

This is engineering automation scaffolding, not clinical validation, compliance certification, emergency-service coverage, or healthcare production readiness.

## Local channel drill

Run `python scripts/run_automation_channel_drill.py` to exercise 30 signed
webhook deliveries through a localhost HTTP receiver. The receiver verifies
the HMAC envelope, enforces the redacted payload contract, emits a signed
synthetic delivery receipt, and records local p50/p95 latency in
`Data/evals/ops/latest_automation_channel_drill.json`.

This is a real local protocol round trip, but it is not a live n8n, email, SMS,
or Viber delivery. The recipient is synthetic, no PHI is allowed, and a receipt
is explicitly not clinician acknowledgement.
