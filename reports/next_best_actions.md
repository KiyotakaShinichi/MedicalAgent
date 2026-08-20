# Next Best Actions

## Task ID: NLC-001
- Rank: 1
- Severity: Critical
- Domain: Authentication / Security
- Why now: Without a real, validated identity path, the project still cannot support any authenticated deployment beyond demo mode.
- Problem: Demo auth is local-only and not sufficient for staging or production. Real OIDC and session controls are still unproven.
- Evidence: [backend/services/auth.py](backend/services/auth.py), [backend/services/oidc_auth.py](backend/services/oidc_auth.py), [backend/api/routers/auth.py](backend/api/routers/auth.py)
- Files/modules: backend/services/auth.py; backend/services/oidc_auth.py; backend/api/deps.py; backend/api/routers/auth.py
- Proposed solution: Replace demo-only assumptions with a validated staging IdP integration, strict JWT validation, environment-specific auth gating, and session-expiry tests.
- Acceptance criteria: staging auth works with real tokens; invalid tokens rejected; no demo auth enabled in staging/prod; session rotation verified.
- Tests/evidence required: JWT verification tests, OIDC happy-path tests, replay and expiry tests, production config validation
- Complexity: M
- Deployment risk reduced: high
- Blocks which deployment stage: Stage 2, 3, 4, 5

## Task ID: NLC-002
- Rank: 2
- Severity: Critical
- Domain: Authorization / Tenant Isolation
- Why now: The project contains tenant logic but no proven multi-user isolation.
- Problem: Real multi-user patient-resource access must be proven before any beta or production deployment.
- Evidence: [backend/services/tenant_scoping.py](backend/services/tenant_scoping.py), [backend/services/saas_control_plane.py](backend/services/saas_control_plane.py), [backend/api/deps.py](backend/api/deps.py)
- Files/modules: backend/services/tenant_scoping.py; backend/services/saas_control_plane.py; backend/models.py; backend/api/deps.py
- Proposed solution: Add a strict row-level authorization layer and multi-user access tests for role and patient isolation.
- Acceptance criteria: no cross-tenant or cross-patient access; access matrix is tested; admin/clinician/patient rules enforced.
- Tests/evidence required: tenant isolation matrix, object-level access tests, actor-to-resource tests
- Complexity: M
- Deployment risk reduced: high
- Blocks which deployment stage: Stage 2, 3, 4, 5

## Task ID: NLC-003
- Rank: 3
- Severity: Critical
- Domain: Privacy / Security
- Why now: Privacy lifecycle, retention, deletion, and audit rules are still not operationally proven.
- Problem: The project is synthetic-only by design, but the path to PHI or even real-user data is not yet bounded or operationally tested.
- Evidence: [README.md](README.md), [backend/models.py](backend/models.py), [backend/services/saas_control_plane.py](backend/services/saas_control_plane.py)
- Files/modules: backend/models.py; backend/services/saas_control_plane.py; docs/; README.md
- Proposed solution: Define explicit privacy controls and enforce them in app and deployment policy, including retention and deletion hooks and audit events.
- Acceptance criteria: retention policy for patient data exists; deletion workflow tested; access log audit is documented and enforced.
- Tests/evidence required: retention/deletion tests, audit log verification, complaint handling procedure documentation
- Complexity: M
- Deployment risk reduced: high
- Blocks which deployment stage: Stage 2, 3, 4, 5

## Task ID: NLC-004
- Rank: 4
- Severity: Critical
- Domain: Architecture / Safety
- Why now: The safety system is too fragmented for a credible deployment-grade runtime.
- Problem: There are many overlapping safety and evaluation modules; there is no single authoritative runtime truth yet.
- Evidence: [backend/services/agent_rag.py](backend/services/agent_rag.py), [backend/services/agent_safety.py](backend/services/agent_safety.py), [backend/services](backend/services)
- Files/modules: backend/services/agent_rag.py; backend/services/agent_safety.py; backend/services/agent_intent_router.py; backend/services/dep001* modules
- Proposed solution: Freeze the experimental safety expansion, keep the authoritative runtime path minimal, and archive the duplicates.
- Acceptance criteria: one runtime safety path, documented ownership, explicit fail-closed tests, no conflicting policy branches.
- Tests/evidence required: route and fail-closed tests, policy drift verification, safety path invariants
- Complexity: M
- Deployment risk reduced: high
- Blocks which deployment stage: Stage 2, 3, 4, 5

## Task ID: NLC-005
- Rank: 5
- Severity: Critical
- Domain: Reliability / Operations
- Why now: Production reliability is still not demonstrated under dependency failure.
- Problem: Cache, DB, vector store, queue, and recovery failure modes are not evidenced under realistic operational conditions.
- Evidence: [backend/services/container_recovery_smoke.py](backend/services/container_recovery_smoke.py), [docker-compose.recovery-smoke.yml](docker-compose.recovery-smoke.yml), [backend/services/agent_cache.py](backend/services/agent_cache.py)
- Files/modules: backend/services/container_recovery_smoke.py; backend/services/agent_cache.py; docker-compose.recovery-smoke.yml; backend/api/main.py
- Proposed solution: Run real recovery and failover drills across the system’s critical dependencies and codify policy for each dependency failure.
- Acceptance criteria: each critical dependency has a documented fail-closed or fail-safe policy; recovery test passes.
- Tests/evidence required: fault injection tests, DB outage simulation, queue backlog test, cache corruption recovery test, rollback rehearsal
- Complexity: M
- Deployment risk reduced: high
- Blocks which deployment stage: Stage 2, 3, 4, 5

## Task ID: NLC-006
- Rank: 6
- Severity: High
- Domain: Observability
- Why now: Without a proper incident workflow, operators cannot safely diagnose a production issue.
- Problem: There are telemetry and request IDs, but not enough operator-grade evidence to act under downtime or incorrect behavior.
- Evidence: [backend/api/main.py](backend/api/main.py), [backend/services/request_context.py](backend/services/request_context.py), [backend/services/llm_telemetry.py](backend/services/llm_telemetry.py)
- Files/modules: backend/api/main.py; backend/services/request_context.py; backend/services/llm_telemetry.py
- Proposed solution: Add an operational dashboard and alerting baseline with key metrics for latency, failures, retrieval behavior, and safety outcomes.
- Acceptance criteria: operator can identify request path, failure reason, and impact volume from logs and metrics.
- Tests/evidence required: trace replay, alert rules, incident drill
- Complexity: M
- Deployment risk reduced: medium-high
- Blocks which deployment stage: Stage 2, 3

## Task ID: NLC-007
- Rank: 7
- Severity: High
- Domain: Deployment / DevOps
- Why now: The stack needs a realistic private staging environment, not more demo scaffolding.
- Problem: The repo has local Docker and readiness artifacts, but not a proven production-shaped private staging plan.
- Evidence: [docker-compose.yml](docker-compose.yml), [docker-compose.prod.yml](docker-compose.prod.yml), [backend/services/deployment_readiness.py](backend/services/deployment_readiness.py)
- Files/modules: docker-compose.yml; docker-compose.prod.yml; backend/services/deployment_readiness.py; .github/workflows/*
- Proposed solution: Establish a private staging environment with TLS, auth, health checks, and explicit deployment rollback path.
- Acceptance criteria: staging deployment is reproducible; health checks pass; rollback rehearsed.
- Tests/evidence required: smoke tests, deployment checks, rollback verification
- Complexity: M
- Deployment risk reduced: medium-high
- Blocks which deployment stage: Stage 2, 3

## Task ID: NLC-008
- Rank: 8
- Severity: High
- Domain: Evaluation Governance
- Why now: DEP-001 should not continue consuming engineering time without a clear deployment gain.
- Problem: The repository is over-invested in internal evaluation loops without external proof.
- Evidence: [config/release_gate_thresholds.yaml](config/release_gate_thresholds.yaml), [README.md](README.md), [Data/evals/governance/latest_focused_release_summary.json](Data/evals/governance/latest_focused_release_summary.json)
- Files/modules: config/release_gate_thresholds.yaml; Data/evals; backend/services/dep001*
- Proposed solution: Freeze active DEP-001 work, keep historical evidence, and require a new external holdout before further evaluation is prioritized.
- Acceptance criteria: no new DEP-001 tuning loops until blockers are addressed; external holdout or fresh proof required.
- Tests/evidence required: governance review, freeze record, external evidence plan
- Complexity: S
- Deployment risk reduced: medium
- Blocks which deployment stage: Stage 3, 4, 5

## Task ID: NLC-009
- Rank: 9
- Severity: High
- Domain: Database / Recovery / Infra
- Why now: Backups, restore drills, and migration safety are still not proven.
- Problem: The project cannot yet claim safe production operations if database and config rollback are not proven.
- Evidence: [docker-compose.yml](docker-compose.yml), [backend/database.py](backend/database.py), [backend/services/container_recovery_smoke.py](backend/services/container_recovery_smoke.py)
- Files/modules: backend/database.py; docker-compose.yml; docker-compose.recovery-smoke.yml; backend/services/container_recovery_smoke.py
- Proposed solution: Implement a real backup and restore drill with migration rollback validation in staging.
- Acceptance criteria: restore from backup passes, migration rollback documented, operator playbook exists.
- Tests/evidence required: backup restore drill and rollback rehearsal
- Complexity: M
- Deployment risk reduced: medium-high
- Blocks which deployment stage: Stage 2, 3

## Task ID: NLC-010
- Rank: 10
- Severity: Medium
- Domain: Product / UX / Safety Boundaries
- Why now: A controlled beta still needs a tight product boundary and user-facing guardrails.
- Problem: The system is a medical monitoring prototype, but product UX must clearly constrain patient-facing behavior and avoid unsafe assumptions.
- Evidence: [README.md](README.md), [backend/services/agent_rag.py](backend/services/agent_rag.py), [frontend-react](frontend-react)
- Files/modules: frontend-react; backend/services/agent_rag.py; backend/services/agent_safety.py; README.md
- Proposed solution: Add explicit UX guardrails, agent behavior disclaimers, and review-only boundaries in the product flow.
- Acceptance criteria: product content and UI warnings clearly distinguish supported education from unsafe or unsupported output.
- Tests/evidence required: UX safety review, route behavior tests, patient-education boundary tests
- Complexity: S
- Deployment risk reduced: medium
- Blocks which deployment stage: Stage 3, 4
