# n8n + Pinecone Industry Integration Readiness

n8n and Pinecone integration readiness is software architecture planning only. It does not make NLCare clinically validated, HIPAA compliant, production healthcare ready, clinician-approved, or safe for real patient care. External workflow automation and managed vector search must remain optional and disabled for patient-specific or PHI workflows until compliance, security review, and clinical governance exist.

## Recommended Order

1. n8n for internal evaluation/review automation only
2. Pinecone shadow index for synthetic/demo KB retrieval only
3. dual-run FAISS/BM25/Pinecone retrieval comparison
4. optional admin-only retrieval diagnostics
5. compliance/security review before any real patient data or PHI

## n8n

- Role: `internal_workflow_automation`
- Status: `optional_disabled_by_default`

Recommended uses:
- release-gate result notification to Discord/Slack/email
- external reviewer intake and attestation reminders
- scheduled eval refresh workflow for non-live synthetic/internal artifacts
- admin-only incident ticket creation when unsafe leakage or stale blockers appear
- dataset integration checklist tracking for BreastDCEDL, Duke MRI, GENIE BPC, MIMIC-IV, and ClinVar

Not allowed uses:
- patient-facing clinical advice
- automatic clinical escalation without human review
- treatment or dosage decisions
- genetic counseling or VUS interpretation
- tumor-marker conclusion workflow
- PHI workflow before compliance review

## Pinecone

- Role: `optional_managed_vector_backend_shadow_mode`
- Status: `optional_disabled_by_default`

Recommended uses:
- shadow retrieval comparison against FAISS/BM25 on synthetic/demo KB
- managed namespace experiments for source-tier governance
- metadata-filter stress testing for source_tier, allowed_use, patient_facing, and kb_fingerprint
- latency/cost comparison artifact before any promotion

Not allowed uses:
- raw patient chat or PHI storage
- replacement of source-tier filtering
- replacement of claim validation
- patient-specific memory before compliance review
- clinical confidence scoring

Namespace plan:
- `nlcare_kb_demo_t1_t3`: patient-facing synthetic/demo KB chunks only
- `nlcare_eval_synthetic`: frozen eval chunks and synthetic test fixtures
- `nlcare_clinician_only_shadow`: disabled by default; clinician-only docs never cited to patient-facing routes
- `patient_data`: disallowed until compliance/security review

## Acceptance Checks Before Live Use

- External services disabled by default in local and demo configs.
- No PHI or patient-specific chat turns sent to n8n or Pinecone.
- Pinecone retrieval must preserve same source-tier filtering, allowed-use filtering, staleness checks, and citation validation.
- n8n workflows may trigger evals, review intake, and admin alerts; they may not issue medical advice or treatment actions.
- All outbound requests carry request IDs and redact patient identifiers.
- Rate limiting, retry/backoff, timeout, and audit logging exist before shadow mode.
- A security/compliance review is required before real patient data.

## Official Docs

- https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.webhook/
- https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-langchain.chattrigger/
- https://docs.pinecone.io/guides/index-data/data-modeling
- https://docs.pinecone.io/guides/search/filter-by-metadata
- https://docs.pinecone.io/guides/index-data/upsert-data
