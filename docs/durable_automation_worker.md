# Durable automation worker

NLCare stores approved, redacted engineering jobs in `async_tasks`. A separate
worker claims one row with a database lease, heartbeats while it runs, persists
the result, schedules bounded retries, and recovers leases left behind by a
crashed process.

```powershell
# Process one queued job
.\.venv\Scripts\python.exe scripts\run_automation_worker.py --once

# Poll continuously (run under a process supervisor for deployment experiments)
.\.venv\Scripts\python.exe scripts\run_automation_worker.py
```

Webhook jobs use HMAC-signed n8n envelopes. Signed callback receipts are stored
against either the high-risk review alert or automation task that emitted the
event. A receipt means only that a delivery channel reported `accepted`,
`delivered`, or `failed`; it is never treated as clinician acknowledgement,
patient contact, review completion, emergency coverage, or clinical action.

The worker remains an engineering prototype. Live high-risk delivery is limited
to an explicitly configured synthetic test recipient, PHI fields are blocked,
and patient-facing clinical decisions are outside the job allowlist.
