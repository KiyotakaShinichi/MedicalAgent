# NLCare Deployment Tasks

Tasks are ordered by safety, security/privacy, fail-closed reliability, deployment blockers, observability, maintainability, performance, then polish. The safe scope remains synthetic and education-only.

## DEP-001 - Independent Critical-Intent Final-Output Evaluation

- **Priority:** P0
- **Severity:** Critical
- **Status:** BLOCKED - frozen model-authored holdout failed routing/generalization; external human author still pending.
- **Domain:** Medical safety / AI evaluation
- **Problem:** Held-out v2 pass rate is `0.7818` with reported unsafe-intent leakage `0.2182`; current cases are not independently authored.
- **Evidence:** `Data/evals/safety/latest_adversarial_generalization_v2_eval.json`; `docs/review_packets/external_author_eval_packet.md`.
- **Exact files/modules involved:** `backend/services/unsafe_intent_semantic_classifier.py`, final response validators, adversarial runners, `Data/evals/safety/`.
- **Proposed implementation:** Have an eligible no-read author create multi-turn critical-family cases; measure routing and final released output separately; adjudicate disputed labels without tuning on the holdout.
- **Acceptance criteria:** At least 20 cases per critical family where feasible; contamination attestation; zero treatment/dose/prognosis/genetic/tumor-marker authority in final outputs; confidence intervals and failures published.
- **Tests required:** Frozen-hash test, contamination metadata test, final-output leakage scorer, safe-negative over-refusal test, Taglish/code-switch parity.
- **Dependencies:** Independent author; later clinician adjudication for ambiguous labels.
- **Estimated complexity:** L
- **Deployment risk reduced:** Prevents unseen phrasing from bypassing boundaries.
- **Blocks deployment:** Yes, Stage 3+.
- **Latest reassessment:** The 180-case frozen no-read model-authored bank had
  zero released unsafe candidate outputs (0/110) and passed 8/8 fault
  injections, but unsafe-intent recall was 0.3727, urgent escalation recall was
  0.2000, over-refusal was 0.2200, and EN/Taglish parity was 0.8063. DEP-001 is
  not complete. Do not tune against this bank; remediate from separate
  development data, then commission a fresh eligible external-human holdout.
- **DEP-001A remediation (2026-08-13):** Added a frozen multilingual encoder,
  calibrated unsafe and urgent heads, structured turn-level risk state,
  fail-closed artifact loading, and independent post-generation containment.
  On the new internal 1,150-case validation bank the layered router has zero
  unsafe passes, urgent recall `0.98`, Taglish unsafe recall `1.0`,
  EN/Taglish gap `0.0126`, multi-turn recall `1.0`, and over-refusal `0.04`;
  12/12 fault injections pass. These are internally generated engineering
  results. DEP-001 remains blocked pending a newly authored external-human
  no-read holdout after this implementation freeze.

## DEP-002 - Live OIDC and Tenant/Care-Team Authorization

- **Priority:** P0
- **Severity:** High
- **Domain:** Authentication / authorization
- **Problem:** Demo credentials and global clinician/admin scopes are not acceptable for network deployment.
- **Evidence:** `backend/services/auth.py:14-19,81,87-97`; `backend/api/deps.py`; OIDC scaffolding has no live-provider evidence.
- **Exact files/modules involved:** `backend/services/oidc_auth.py`, `backend/api/deps.py`, auth router, frontend auth context, organization/membership models.
- **Proposed implementation:** Configure one real OIDC tenant with PKCE; map immutable subject to server-owned organization/care-team membership; deny header-selected tenant context; define session revocation and role-change behavior.
- **Acceptance criteria:** Demo login disabled in staging; forged role/tenant claims rejected; patient object isolation and clinician assignment enforced; access audit exported.
- **Tests required:** Live-provider sandbox test, expired/revoked token, wrong audience/issuer, cross-tenant matrix, role downgrade, concurrent logout.
- **Dependencies:** Identity provider and test tenants.
- **Estimated complexity:** XL
- **Deployment risk reduced:** Prevents account and cross-tenant data compromise.
- **Blocks deployment:** Yes, Stage 2+.

## DEP-003 - Privacy Lifecycle and Data Inventory

- **Priority:** P0
- **Severity:** High
- **Domain:** Privacy / governance
- **Problem:** Retention, export, deletion, consent, key ownership, and privacy incident workflows are not operationally implemented.
- **Evidence:** Local DB/uploads/traces; existing limitation docs; no tested delete/export endpoint or retention worker.
- **Exact files/modules involved:** patient models, upload storage, chat/RAG logs, audit logs, caches, object storage adapter, `docs/privacy*`, `docs/incident*`.
- **Proposed implementation:** Create a machine-readable data inventory and retention classes; implement synthetic account export/delete with tombstones and audit receipts; purge cache/uploads/traces; document backup deletion limitations.
- **Acceptance criteria:** Every stored field has purpose/owner/retention; export is complete; deletion removes active copies and records a non-sensitive receipt; restoration does not silently resurrect deleted data.
- **Tests required:** End-to-end export/delete, cache purge, object-store lifecycle, backup restore with deletion ledger, logging redaction.
- **Dependencies:** Object storage and identity design.
- **Estimated complexity:** XL
- **Deployment risk reduced:** Reduces unauthorized retention and privacy failure.
- **Blocks deployment:** Yes, Stage 3+.

## DEP-004 - Path-Confinement for Admin Data and Model Workflows

- **Priority:** P0
- **Severity:** High
- **Domain:** Security / backend
- **Problem:** Admin routes accept host input/output paths for datasets, models, manifests, previews, and XAI.
- **Evidence:** Request models in `backend/api/routers/model.py:26-147`. CSV `file_path` was disabled in this reassessment, but other admin paths remain.
- **Exact files/modules involved:** model router, CSV/MRI/BreastDCEDL services, artifact writers.
- **Proposed implementation:** Replace arbitrary paths with artifact IDs or paths resolved under configured import/output roots using `Path.resolve()` containment; reject symlinks and absolute/out-of-root paths.
- **Acceptance criteria:** No network payload can read/write outside approved roots; writes are atomic; artifact IDs and hashes are audited.
- **Tests required:** `..`, absolute path, symlink/junction, alternate drive, UNC, race, and valid-root cases.
- **Dependencies:** Object/artifact storage contract.
- **Estimated complexity:** L
- **Deployment risk reduced:** Prevents host file disclosure/overwrite by compromised admin.
- **Blocks deployment:** Yes, Stage 2.

## DEP-005 - Queue All Heavy Model/Index/Data Jobs

- **Priority:** P0
- **Severity:** High
- **Domain:** Reliability / backend
- **Problem:** Training, preprocessing, indexing, and generation execute synchronously inside API workers.
- **Evidence:** `backend/api/routers/model.py:208-554`.
- **Exact files/modules involved:** model router, `backend/services/automation_job_queue.py`, worker runner, experiment tracking.
- **Proposed implementation:** API validates and enqueues idempotent job; worker executes with resource/time limits; status/result references returned; cancellation and dead-letter supported.
- **Acceptance criteria:** API responds in <500 ms for submission; duplicate idempotency key reuses job; worker crash recovers once; no duplicate registry promotion.
- **Tests required:** burst submission, worker kill, retry, cancellation, timeout, duplicate, poison job, resource limit.
- **Dependencies:** DEP-004 and durable artifact storage.
- **Estimated complexity:** L
- **Deployment risk reduced:** Prevents API starvation and duplicate/corrupt work.
- **Blocks deployment:** Yes, Stage 2.

## DEP-006 - Blocking Container Security and Runtime Identity

- **Priority:** P0
- **Severity:** High
- **Domain:** Security / supply chain
- **Problem:** Current artifact reports 15 High CVEs and `BLOCK_PUBLIC_DEPLOYMENT`; non-root execution was not verified.
- **Evidence:** `Data/evals/ops/latest_container_security_scan.json`.
- **Exact files/modules involved:** `Dockerfile`, scan scripts, CI/ship workflows, `config/dependency_risk_acceptance.json`.
- **Proposed implementation:** Rebuild against current digest, verify configured/running UID, scan exact image digest, attach SBOM/provenance, require time-bounded owner-approved risk acceptance only when no fix exists.
- **Acceptance criteria:** Zero Critical and zero unaccepted High; scanned digest equals deployed digest; non-root verified; SBOM attached.
- **Tests required:** image user assertion, filesystem write boundary, capability check, exact-digest scan, expiry test.
- **Dependencies:** Container registry.
- **Estimated complexity:** M
- **Deployment risk reduced:** Reduces exploitable runtime/supply-chain exposure.
- **Blocks deployment:** Yes, Stage 2+.

## DEP-007 - Alembic-Only Strict-Environment Schema Ownership

- **Priority:** P0
- **Severity:** High
- **Domain:** Database / deployment
- **Problem:** App import executes `ensure_schema()` and runtime DDL while entrypoint also runs Alembic.
- **Evidence:** `backend/api/main.py:69`; `backend/schema_migrations.py:6-110`; `scripts/container_entrypoint.py`.
- **Exact files/modules involved:** those files and migration tests.
- **Proposed implementation:** In staging/production, run migrations as a predeploy job; app startup only verifies current head. Keep legacy patcher for local SQLite tools only.
- **Acceptance criteria:** API DB role lacks DDL; startup fails on wrong head; concurrent replicas perform no DDL; empty and upgrade paths succeed.
- **Tests required:** Postgres empty migration, N-1 upgrade, wrong-head startup, concurrent startup, rollback/forward recovery.
- **Dependencies:** Production-compose integration environment.
- **Estimated complexity:** M
- **Deployment risk reduced:** Prevents race, partial schema, and privileged app runtime.
- **Blocks deployment:** Yes, Stage 2.

## DEP-008 - Managed Backup, Restore, and Disaster-Recovery Drill

- **Priority:** P0
- **Severity:** High
- **Domain:** Infrastructure / reliability
- **Problem:** Volumes exist but RPO/RTO, PITR, encryption, restore, and deletion behavior are unproven.
- **Evidence:** `docker-compose.prod.yml`; synthetic resilience artifacts remain local/external-blocked.
- **Exact files/modules involved:** infrastructure templates, runbooks, DB/object-store configuration, restore tests.
- **Proposed implementation:** Managed Postgres PITR, versioned object storage, encrypted backups, restore into isolated environment, integrity and deletion-ledger checks.
- **Acceptance criteria:** Documented RPO/RTO; successful timed restore; hashes and migration head match; secrets are not restored from data backups.
- **Tests required:** quarterly restore drill, corrupted backup, missing object, deleted-account non-resurrection.
- **Dependencies:** Private staging cloud account.
- **Estimated complexity:** L
- **Deployment risk reduced:** Prevents irrecoverable or inconsistent loss.
- **Blocks deployment:** Yes, Stage 2.

## DEP-009 - External No-Read RAG Baseline and Goldset Adjudication

- **Priority:** P1
- **Severity:** High
- **Domain:** RAG / evaluation
- **Problem:** Full governed stack trails BM25 and current gold labels conflict with patient-facing source policy.
- **Evidence:** `latest_rag_baseline_comparison.json`; adjudication and holdout readiness artifacts.
- **Exact files/modules involved:** RAG baseline runner, retrieval goldsets, source-tier policies, no-read protocol.
- **Proposed implementation:** Complete independent holdout; adjudicate source-filter-drop cases; compare BM25, hybrid, governed hybrid, and reranker with paired intervals.
- **Acceptance criteria:** Holdout never used for tuning; source labels are audience-compatible; failure examples and negative result visible.
- **Tests required:** hash freeze, contamination attestation, source-policy validator, paired bootstrap/randomization test.
- **Dependencies:** External author/reviewer.
- **Estimated complexity:** M
- **Deployment risk reduced:** Prevents unjustified complex retrieval and unsupported citations.
- **Blocks deployment:** Yes, Stage 3.

## DEP-010 - Canonical Release Decision Surface

- **Priority:** P1
- **Severity:** High
- **Domain:** Evaluation governance
- **Problem:** A 229-artifact pass can coexist with a public-deployment-blocking image scan.
- **Evidence:** `Data/evals/governance/latest_release_gate_run.json`: 28 decision, 201 appendix, 0 failures.
- **Exact files/modules involved:** `config/release_gate_thresholds.yaml`, release runner, registry, dashboard.
- **Proposed implementation:** Create a <=25-item canonical gate for safety, auth, security, migrations, recovery, RAG, and deployment; keep all other artifacts in linked appendices.
- **Acceptance criteria:** Any Critical/High deployment blocker fails relevant stage; every gate has owner, stage, freshness, source independence, and remediation URL.
- **Tests required:** inject stale/missing/blocked artifact, stage-aware gate tests, contradictory-status test.
- **Dependencies:** Agreement on stage policy.
- **Estimated complexity:** M
- **Deployment risk reduced:** Prevents green-by-volume release decisions.
- **Blocks deployment:** Yes, Stage 2.

## DEP-011 - Private Synthetic Staging Foundation

- **Priority:** P1
- **Severity:** High
- **Domain:** Infrastructure / deployment
- **Problem:** No running managed environment proves the reference architecture.
- **Evidence:** Compose/Bicep are repository scaffolds; external managed evidence is blocked.
- **Exact files/modules involved:** `infra/`, `docker-compose.prod.yml`, deployment validation, secrets/config docs.
- **Proposed implementation:** Deploy TLS edge, API, frontend, worker, managed Postgres/Redis/object storage/secrets, and central telemetry in a private network.
- **Acceptance criteria:** No demo auth; no public DB/cache; synthetic-only enforced; immutable image/artifact digests; readiness and alerts verified.
- **Tests required:** infrastructure policy, endpoint exposure scan, secret rotation, deployment smoke, rollback.
- **Dependencies:** DEP-006 to DEP-008.
- **Estimated complexity:** XL
- **Deployment risk reduced:** Replaces local assumptions with operational evidence.
- **Blocks deployment:** Yes, Stage 2.

## DEP-012 - OpenTelemetry, Provider Usage, and SLO Dashboards

- **Priority:** P1
- **Severity:** High
- **Domain:** Observability
- **Problem:** Token/cost figures are estimates and dependency/operator visibility is incomplete.
- **Evidence:** `latest_cost_latency_report.json` has `0.0` actual usage coverage; route report says `production_ready: false`.
- **Exact files/modules involved:** request middleware, LLM telemetry, RAG logs, worker metrics, frontend admin health.
- **Proposed implementation:** Export traces/metrics/logs with privacy-safe IDs; ingest provider token counts; create API/RAG/queue/dependency dashboards and alerts from Section 14.
- **Acceptance criteria:** >=95% provider usage coverage on provider calls; trace coverage >=99%; alert drill links request to API, retrieval, provider, DB, and worker.
- **Tests required:** telemetry schema, dropped sink, sampling, redaction, alert simulation.
- **Dependencies:** Private staging telemetry backend.
- **Estimated complexity:** L
- **Deployment risk reduced:** Enables incident detection, cost control, and diagnosis.
- **Blocks deployment:** Yes, Stage 3; partially Stage 2.

## DEP-013 - Split Liveness from Readiness and Expand Dependency Health

- **Priority:** P1
- **Severity:** Medium
- **Domain:** Reliability
- **Problem:** `/health` queries DB; `/ready` omits Redis, workers, provider circuit, scanner, and artifact freshness.
- **Evidence:** `backend/api/main.py:171-217`.
- **Exact files/modules involved:** API probes, compose health checks, ops health snapshot.
- **Proposed implementation:** Liveness is process-only; readiness is policy-driven with required/optional dependencies; worker has separate heartbeat probe.
- **Acceptance criteria:** DB outage does not restart healthy process; readiness returns 503 and reason codes; optional LLM outage does not enable unsafe answer fallback.
- **Tests required:** dependency matrix and container restart-behavior test.
- **Dependencies:** Staging compose.
- **Estimated complexity:** M
- **Deployment risk reduced:** Prevents restart storms and false readiness.
- **Blocks deployment:** No for Stage 1; yes for Stage 2.

## DEP-014 - Streaming Request Limits and Cost-Aware Rate Policies

- **Priority:** P1
- **Severity:** Medium
- **Domain:** API security
- **Problem:** Body cap relies on `Content-Length`; LLM chat defaults to 120/min; tenant key uses caller header.
- **Evidence:** `backend/services/api_protection.py:101-168`.
- **Exact files/modules involved:** API middleware, proxy config, auth context, rate limiter.
- **Proposed implementation:** Enforce edge and streaming byte caps; key by authenticated subject/tenant; use token/cost quotas and endpoint concurrency semaphores.
- **Acceptance criteria:** Chunked oversized body rejected; forged org header has no effect; costly routes bounded per user/tenant.
- **Tests required:** chunked upload, spoofed header, multi-replica Redis, provider-cost burst.
- **Dependencies:** OIDC/tenant context and reverse proxy.
- **Estimated complexity:** M
- **Deployment risk reduced:** Limits denial of service and spend abuse.
- **Blocks deployment:** Yes, Stage 3.

## DEP-015 - Immutable Signed Model, KB, and Index Bundles

- **Priority:** P1
- **Severity:** High
- **Domain:** MLOps / RAG governance
- **Problem:** Serving uses mutable mounted artifacts and registry metadata without a signed deployment bundle.
- **Evidence:** `docker-compose.prod.yml` mounts `Data` and `KnowledgeBase`; existing hashes are not a startup authorization package.
- **Exact files/modules involved:** model registry, KB fingerprint, vector index loader, deployment config, object storage adapter.
- **Proposed implementation:** Build signed content-addressed bundle manifests and verify code/model/schema/calibration/KB/index compatibility at startup.
- **Acceptance criteria:** Bit-flip or mismatched fingerprint fails readiness; rollback selects an immutable predecessor; actor and approval logged.
- **Tests required:** tamper, mismatch, expired approval, missing bundle, rollback.
- **Dependencies:** Artifact storage and signing key management.
- **Estimated complexity:** L
- **Deployment risk reduced:** Prevents poisoned/stale/mismatched artifacts.
- **Blocks deployment:** Yes, Stage 3.

## DEP-016 - Multi-Replica and Chaos Qualification

- **Priority:** P1
- **Severity:** High
- **Domain:** Scalability / reliability
- **Problem:** Multi-process cache/index consistency and managed failure behavior are not proven.
- **Evidence:** Current compose runs one backend; local-only resilience artifacts.
- **Exact files/modules involved:** cache, vector runtime, workers, DB leases, load/fault scripts.
- **Proposed implementation:** Run two API replicas and multiple workers; inject Redis/DB/provider/network failures; verify no cross-user cache leak or duplicate action.
- **Acceptance criteria:** SLOs hold at 100 concurrent synthetic users; all failures are bounded/fail-closed; recovery time measured.
- **Tests required:** soak, spike, network partition, worker kill, index rollover, cache invalidation.
- **Dependencies:** DEP-011 and DEP-012.
- **Estimated complexity:** L
- **Deployment risk reduced:** Exposes distributed-state defects before beta.
- **Blocks deployment:** Yes, Stage 3.

## DEP-017 - Upload Storage and Lifecycle Hardening

- **Priority:** P1
- **Severity:** Medium
- **Domain:** Security / data engineering
- **Problem:** Upload validation is strong for a prototype, but storage is local and lifecycle/CDR/restore behavior is incomplete.
- **Evidence:** `backend/services/upload_security.py`; patient upload service; uploads disabled in strict compose.
- **Exact files/modules involved:** upload router/service, scanner adapter, object storage, retention worker.
- **Proposed implementation:** Quarantine bucket, external scanner/CDR, encrypted object store, presigned download, ownership policy, lifecycle deletion.
- **Acceptance criteria:** Strict environment cannot enable uploads without scanner/storage; no local executable path; all access audited.
- **Tests required:** polyglot, archive bomb policy, malicious PDF, scanner outage, cross-user download, deletion.
- **Dependencies:** Object storage and privacy lifecycle.
- **Estimated complexity:** L
- **Deployment risk reduced:** Contains malicious content and data exposure.
- **Blocks deployment:** Only if uploads enabled.

## DEP-018 - CI Runtime Parity and Production-Compose Integration

- **Priority:** P1
- **Severity:** Medium
- **Domain:** DevOps / testing
- **Problem:** CI Python 3.11 differs from serving Python 3.13; image build is not followed by a production-compose migration/recovery test.
- **Evidence:** `.github/workflows/ci.yml`, `Dockerfile`.
- **Exact files/modules involved:** CI/ship workflows, lockfiles, compose, migration smoke.
- **Proposed implementation:** Test supported Python version matrix or align to image; run exact image with Postgres/Redis, migrations, workers, smoke, and teardown.
- **Acceptance criteria:** Exact deployed image digest passes integration; generated types clean; no mutable dependency resolution.
- **Tests required:** compose integration, migration, worker, OIDC stub, readiness, rollback smoke.
- **Dependencies:** CI capacity.
- **Estimated complexity:** M
- **Deployment risk reduced:** Prevents environment-specific release failures.
- **Blocks deployment:** Yes, Stage 2.

## DEP-019 - Canonical Package Ownership and Dead-Code Reduction

- **Priority:** P2
- **Severity:** Medium
- **Domain:** Software engineering
- **Problem:** 417 service modules, 348 scripts, compatibility wrappers, and 378 eval artifacts create hidden coupling and review burden.
- **Evidence:** Repository inventory from this reassessment.
- **Exact files/modules involved:** `backend/services/`, `scripts/`, eval registry, ADR index.
- **Proposed implementation:** Ownership map, import graph, deprecation manifest, canonical package boundaries, duplicate helper consolidation.
- **Acceptance criteria:** No circular imports across domains; deprecated modules have removal dates; canonical public interfaces documented.
- **Tests required:** import graph/cycle test, dead entrypoint scan, registry referential integrity.
- **Dependencies:** None.
- **Estimated complexity:** XL
- **Deployment risk reduced:** Reduces maintenance regressions and opaque fallbacks.
- **Blocks deployment:** No.

## DEP-020 - Human-Factors and Medical Wording Review

- **Priority:** P1
- **Severity:** High
- **Domain:** Medical safety / product
- **Problem:** Disclaimers and explanations are internally authored; users can still over-trust scores, probabilities, and "care-team review" language.
- **Evidence:** Review packets remain unreviewed; external readiness status is preparation only.
- **Exact files/modules involved:** patient dashboard, chat templates, XAI cards, refusal/escalation templates, review packets.
- **Proposed implementation:** Structured review by oncology nurse/clinician, genetic counselor, pharmacist, and non-expert users; log severity and fixes without implying approval.
- **Acceptance criteria:** Every critical wording surface reviewed; comprehension tasks distinguish monitoring index from health score and delivery receipt from acknowledgement.
- **Tests required:** moderated usability script, comprehension thresholds, accessibility/E2E regression.
- **Dependencies:** Human reviewers.
- **Estimated complexity:** M
- **Deployment risk reduced:** Reduces overtrust and harmful interpretation.
- **Blocks deployment:** Yes, Stage 3.

## DEP-021 - Root License and Documentation Freshness Policy

- **Priority:** P2
- **Severity:** Medium
- **Domain:** Documentation / governance
- **Problem:** No root software license was found; many generated artifacts/docs have no owner or expiry.
- **Evidence:** Repository root and 82-day-old supporting artifacts in the release run.
- **Exact files/modules involved:** root `LICENSE`, README, benchmark registry, documentation index.
- **Proposed implementation:** Select a license with owner approval; add document owner, last verified commit/date, expiry, and canonical/appendix status.
- **Acceptance criteria:** License present; stale canonical evidence fails stage gate; one evidence index links current decisions.
- **Tests required:** license presence, documentation link and freshness checks.
- **Dependencies:** Project owner decision.
- **Estimated complexity:** S
- **Deployment risk reduced:** Clarifies legal reuse and reduces stale-evidence decisions.
- **Blocks deployment:** License blocks public distribution; freshness blocks Stage 2 evidence.

## DEP-022 - Patient-Safe ML/XAI Presentation Gate

- **Priority:** P2
- **Severity:** Medium
- **Domain:** ML / XAI / medical safety
- **Problem:** Synthetic probabilities and monitoring indices can be mistaken for personal outcome or clinical risk.
- **Evidence:** Synthetic-only XAI artifacts and current dashboard model cards.
- **Exact files/modules involved:** hybrid prediction, XAI envelopes, patient dashboard cards, comprehension dossier.
- **Proposed implementation:** Default patient surface to qualitative evidence sufficiency and missingness; place raw probabilities in admin/reviewer view; add "why unavailable" and safe next-step language.
- **Acceptance criteria:** Non-expert test users correctly state that the number is synthetic/nonclinical and cannot guide treatment.
- **Tests required:** UI content contract, longest-text/mobile screenshots, comprehension study.
- **Dependencies:** DEP-020.
- **Estimated complexity:** M
- **Deployment risk reduced:** Reduces automation bias and score overinterpretation.
- **Blocks deployment:** Yes, Stage 3 if numeric outputs remain patient-visible.

## DEP-023 - Bounded Cost and Accuracy-Latency-Cost Policy

- **Priority:** P2
- **Severity:** Medium
- **Domain:** AI operations / cost
- **Problem:** Current cost report is estimate-only and historical tail latency is extreme.
- **Evidence:** `latest_cost_latency_report.json`; `latest_route_latency_budget.json`.
- **Exact files/modules involved:** LLM telemetry, route policy, cache, iterative RAG, AI Trinity artifact.
- **Proposed implementation:** Record provider usage, set route-specific call/token/time budgets, short-circuit deterministic routes, cap retrieval iterations, and reject variants that weaken safety/governance.
- **Acceptance criteria:** >=95% usage coverage; per-route p95 and cost budget; no accuracy/safety regression; cache policy remains privacy-safe.
- **Tests required:** provider reconciliation, budget exhaustion, circuit breaker, cache isolation, paired quality/latency/cost eval.
- **Dependencies:** DEP-012.
- **Estimated complexity:** M
- **Deployment risk reduced:** Prevents unbounded spend and timeout cascades.
- **Blocks deployment:** Yes, Stage 4; warning for Stage 2/3.
