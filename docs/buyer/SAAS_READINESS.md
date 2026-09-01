# SaaS readiness, maturity, and roadmap

## Capability classification

| Capability | State | Evidence / gap |
|---|---|---|
| Local reproducibility | READY | Bootstrap, pinned locks, disposable synthetic demo |
| Fresh-clone/offline verification | READY | Declared provisioning and hermetic test contract |
| Frontend/backend application | READY for research demo | Role portals and API; not clinical production |
| CI and Docker | READY for engineering | Green R4 CI and production-shaped synthetic compose |
| RAG and agent framework | RESEARCH_ONLY | Advanced controls; retrieval and held-out claims remain bounded |
| Safety | RELEASE_BLOCKED | DEP-001 official behavioral failures |
| Synthetic ML/MLE/XAI | RESEARCH_ONLY | Strong lifecycle evidence; no real-world validation |
| Authentication | PARTIAL | Demo sessions plus OIDC seam; no production IdP deployment proof |
| Authorization/tenant keys | PARTIAL | Route guards and scoped keys; complete hostile isolation not proven |
| Database migration | READY for handoff | Alembic chain and PostgreSQL migration tests |
| Backup/restore | PARTIAL | Safe local demo backup; hosted DR absent |
| Privacy lifecycle | PARTIAL | Synthetic boundary/redaction; deletion/export/compliance absent |
| Audit logging | PARTIAL | App events/request IDs; not immutable or centrally retained |
| Observability | PARTIAL | Structured logs, health, local metrics; no durable exporter/vendor adapter |
| Automation | PARTIAL | Durable/signed seams; external delivery operations buyer-owned |
| Hosted deployment | PARTIAL | Staging shape only; TLS/secrets/monitoring/support required |
| Clinical readiness | RELEASE_BLOCKED | No validation, clinician review, real data, IRB, regulatory program |

## SaaS audit

| Area | Classification | Buyer roadmap |
|---|---|---|
| Authentication/authorization/sessions | PARTIAL | Deploy real OIDC, MFA/session policy, authorization audit |
| Multi-user/tenant isolation | PARTIAL | Threat-model and verify every DB/cache/vector/job boundary |
| Database migrations | READY | Add deployment approval and rollback procedure |
| Backup/restore | PARTIAL | Managed backups, encryption, restore drills, RPO/RTO |
| Privacy retention/deletion/export | MISSING | Build jurisdiction-specific lifecycle before non-synthetic data |
| Audit logging | PARTIAL | Immutable centralized sink and access governance |
| Rate limiting | PARTIAL | Redis-backed seam; tune/test buyer traffic profile |
| Secrets | PARTIAL | Environment contract; add buyer vault/KMS and rotation |
| Background tasks | PARTIAL | Leasing/retries exist; add queue/worker SLO operations |
| Email/notifications | OPTIONAL | Buyer-owned provider, consent, delivery and escalation policy |
| Billing/subscriptions | MISSING | Add only if buyer business model requires it |
| Domain/TLS | MISSING | Buyer DNS, certificates, WAF/reverse proxy |
| Admin/support tools | PARTIAL | Evaluation/admin panels exist; support workflow not operated |

## Gap map

### Ready now

- Technical diligence, local synthetic demonstration, source/evidence review,
  offline regression, CI, migration inspection, and deterministic packaging.

### Next for a hosted research SaaS

- Resolve first-party/content licenses; deploy buyer-owned OIDC and secrets;
  independently validate tenant boundaries; add TLS/domain, managed database,
  backup/restore drills, centralized metrics/errors/log retention, support/incident
  ownership, privacy lifecycle, load/soak testing, and fresh independent safety
  evaluation. Billing/email are roadmap choices, not transfer blockers.

### Required before clinical/healthcare production

- Resolve DEP-001; obtain real-world and prospective clinical validation,
  clinician governance, IRB/ethics and regulatory analysis; implement a quality
  system, health-data privacy/security/compliance program, independent security
  testing, human factors, real-user monitoring, incident reporting, and formal
  model/change control. This is a separate productization program.

## Transferability scorecard

| Category | Result |
|---|---|
| Local/fresh-clone reproducibility | READY |
| CI and configuration clarity | READY |
| Deployment/data portability | PARTIAL |
| Asset clarity | READY |
| License clarity | PARTIAL |
| Demo readiness | READY |
| Security boundary | PARTIAL |
| Observability/operations | PARTIAL |
| Handoff | READY |
| SaaS readiness | PARTIAL |
| Clinical readiness | BLOCKED |

## Possible development directions

- Healthcare-AI evaluation platform
- RAG evidence/safety reliability product
- Adversarial healthcare-agent testing service
- Synthetic clinical benchmark platform
- Internal AI-governance tooling
- AI consulting/research accelerator
- White-label medical-AI research framework
- Extension to other specialties after domain redesign and review

These are technical possibilities, not evidence of market demand or guaranteed
businesses.
