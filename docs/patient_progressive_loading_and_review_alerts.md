# Patient Progressive Loading and Review Alerts

## Records-first loading

The patient portal loads `/me/patient-report/core` first. This response contains patient-scoped records, latest CBC, symptoms, timeline, review reasons, medication log, data coverage, and the patient banner. Synthetic model predictions and heavier engineering enrichment are deferred to `/me/patient-report/enrichment`.

The core request schedules one process-local background enrichment job per patient. The UI polls the enrichment endpoint for a bounded period instead of making the HTTP request execute the model synchronously. Cache warming can prepare known demo patients after application startup, and record writes advance a generation token so stale in-flight results are discarded.

On the July 15, 2026 local engineering profile for synthetic patient P001, the records-first response was approximately 0.38 seconds while the old cold synchronous engineering bundle was approximately 11.9 seconds. These are local development measurements, not production latency claims. The UI keeps records visible if enrichment fails.

Patient-report enrichment remains a single-process cache-warming pool. Separately,
admin engineering automation now uses database leases, heartbeats, expiry
recovery, bounded retries, and dead letters through
`scripts/run_automation_worker.py`. This does not make either subsystem a
managed distributed queue; a multi-host deployment still needs a reviewed
queue/database topology, process supervision, cross-worker cache coordination,
and operational load/failure testing.

## Why the 0-100 index is not patient-facing

The previous 0-100 monitoring context index combined simulator-derived model availability and rule-based review deductions. It was not a health score, cancer-status score, treatment-response probability, prognosis, or clinical severity grade. Because a patient could reasonably read it as medical authority, NLCare removed it from patient headlines. The portal now shows the underlying review items, latest records, record coverage, and synthetic model pattern separately. The legacy field remains only for backward-compatible engineering and reviewer surfaces.

## High-priority conversation review outbox

Mortality wording, immediate-danger wording, crisis language, and urgent symptom wording can create a local `high_risk_conversation_alerts` row. The alert references the patient-scoped chat turn but does not duplicate raw message text. Clinician/admin endpoints can list, inspect, and acknowledge these review items.

Optional external notification uses the HMAC-signed n8n adapter and the `high_risk_review_alert` workflow. The outbound event contains only an alert ID, operational priority, event type, sign-in-required review path, delivery scope, and synthetic-recipient scope. It excludes patient ID, name, contact details, and raw conversation text.

The current prototype permits only a synthetic test recipient when external dispatch is explicitly enabled. Real clinician contact channels are deliberately not enabled. Delivery states are separate:

- `disabled`: the review item exists only in NLCare.
- `accepted_by_workflow`: n8n accepted the redacted event.
- `retry_scheduled`: workflow acceptance failed and a bounded retry is due.
- `dead_lettered`: the retry limit was reached; the local review item remains available.
- `accepted_by_channel`: the test channel accepted a signed receipt.
- `delivered_to_channel`: the test channel reported delivery.
- `acknowledged`: a signed-in clinician/admin role explicitly acknowledged the local review item.

Each network attempt is stored in `high_risk_alert_delivery_attempts`. Retry delays use bounded exponential backoff. Signed callbacks are accepted at `/admin/automation/delivery-receipts`; the raw receipt must pass HMAC, timestamp-freshness, redaction, and event/receipt checks. A channel receipt and clinician acknowledgement remain separate facts.

None of these statuses proves that a clinician assessed the patient or took a clinical action. This queue is not an emergency service. Patient-facing urgent guidance tells the user not to wait for a portal response when they feel unsafe or in immediate danger.

## Configuration

```env
N8N_WEBHOOK_DISPATCH_ENABLED=false
N8N_WEBHOOK_BASE_URL=http://127.0.0.1:5678/webhook/nlcare
N8N_WEBHOOK_SIGNING_SECRET=replace_with_a_long_random_secret
NLCARE_ALERT_NOTIFICATION_MAX_ATTEMPTS=3
NLCARE_ALERT_NOTIFICATION_RETRY_BASE_SECONDS=30
NLCARE_ALERT_TEST_RECIPIENT_ONLY=true
NLCARE_PATIENT_ENRICHMENT_PREWARM_ENABLED=true
NLCARE_PATIENT_REPORT_CACHE_TTL_SECONDS=900
NLCARE_PATIENT_CORE_CACHE_TTL_SECONDS=120
NLCARE_ENRICHMENT_WORKERS=1
```

Import the inactive workflow from `Data/evals/ops/n8n_workflow_templates/high_risk_review_alert.json`, configure raw-body HMAC verification and the 300-second replay window, then attach only a synthetic test-recipient node. Configure a signed receipt callback after that node. Do not commit credentials.

Promotion to a real contact channel requires organizational privacy, security, operational-ownership, consent, and clinical-governance review outside this prototype.

## Claim boundary

This is engineering automation and review routing for a synthetic-only prototype. It is not clinical validation, clinician approval, emergency-response coverage, real-world patient safety evidence, or production healthcare readiness.
