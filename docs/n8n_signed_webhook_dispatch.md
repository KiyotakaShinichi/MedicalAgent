# Signed n8n Webhook Dispatch

NLCare emits only redacted engineering events to n8n. The sender builds a canonical JSON envelope and signs it with HMAC-SHA256 using `N8N_WEBHOOK_SIGNING_SECRET`. Nested blocked fields are rejected before signing.

Required environment variables for live dispatch:

- `N8N_WEBHOOK_DISPATCH_ENABLED=true`
- `N8N_WEBHOOK_BASE_URL=https://<n8n-host>/webhook/nlcare`
- `N8N_WEBHOOK_SIGNING_SECRET=<secret stored outside git>`

Plain HTTP is accepted only for `localhost` development. Imported n8n templates remain inactive. Before activation, configure receiver-side HMAC verification against the raw request body and reject missing, invalid, or replayed event IDs. The generated template checks the redacted payload contract; receiver-side cryptographic verification is still an operator configuration step.

Allowed events cover release-gate alerts, stale-artifact tickets, reviewer reminders, eval refresh notifications, trace-quality digests, Pinecone shadow reports, external red-team intake, dependency alerts, and demo deployment health. They cannot carry PHI, raw chat, raw prompts/responses, private chain-of-thought, or clinical instructions.

This is engineering automation scaffolding, not clinical validation, compliance certification, or healthcare production readiness.
