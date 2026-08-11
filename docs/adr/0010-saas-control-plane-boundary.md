# ADR 0010: Separate SaaS control plane from the synthetic patient demo

- Status: accepted
- Date: 2026-08-10

## Context

NLCare already contains a patient/clinician demonstration, evaluation pipelines,
workers, automation, and governance artifacts. Turning those pieces directly into
a multi-tenant product would mix synthetic patient-demo records with customer,
project, usage, and job-control data. It would also encourage clinical-product
claims that the evidence does not support.

## Decision

Add a separate SaaS control-plane domain for organizations, memberships,
projects, synthetic environments, entitlements, usage events, evaluation jobs,
outbox events, and audit events. Every project, job, usage query, and mutation is
organization scoped. Jobs and outbox delivery use leases, idempotency keys,
bounded retries, crash recovery, and dead-letter states.

The product surface is an **AI assurance and evaluation workspace for synthetic
engineering workflows**. The existing patient portal remains an isolated demo.
It is not silently reclassified as a deployable clinical application.

External n8n delivery remains disabled by default. When enabled, the outbox may
send only signed, redacted control-plane metadata. Real patient data, raw prompts,
raw responses, clinical conclusions, and billing authority are forbidden.

## Consequences

- Tenant isolation becomes explicit and testable for the new control plane.
- The legacy patient-demo tables are not yet tenant migrated and cannot be
  offered as a multi-tenant clinical service.
- Usage limits are engineering entitlements, not invoicing or audited billing.
- Local and production-shaped Compose profiles include separate evaluation and
  outbox workers, but managed-cloud execution is still unproven.
- A real SaaS launch still requires verified OIDC, managed Postgres migration and
  recovery drills, security review, secrets/TLS/gateway controls, and external
  evaluation.
