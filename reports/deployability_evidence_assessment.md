# NLCare Deployability and Evidence Assessment

## Scope and Decision Boundary

This assessment separates three targets that must not be collapsed into one
readiness score:

1. **Local synthetic engineering demo:** no real patient data, no external
   delivery, and no claim of clinical use.
2. **Restricted synthetic staging:** networked deployment with synthetic data,
   production-shaped identity, secrets, database, queue, monitoring, recovery,
   and explicit data-entry restrictions.
3. **Patient-facing healthcare deployment:** real health data and users. This
   remains blocked by safety generalization, privacy, security, clinical review,
   operational ownership, and validation gaps.

The project can move substantially closer to targets 1 and 2 inside the current
constraints. It cannot become ready for target 3 through additional internal
tests or synthetic artifacts alone.

## Current Assessment After This Hardening Pass

| Area | Engineering evidence now | Main unresolved evidence gap | Readiness judgment |
|---|---|---|---|
| AI engineering | Typed final evidence envelope, deny-by-default authorization, exact-response digest binding, route/cache/stream fault injection | Independent end-to-end safety evidence and validator correctness | Strong internal engineering; not patient-ready |
| RAG | Dense/sparse/RRF, source governance, uncertainty routing, frozen comparisons, negative-result reporting | Lexical claim validation, no proven raw lift over BM25, external no-read holdout incomplete | Governed but not independently proven |
| Agentic workflows | Bounded tools, confirmation, verifier checks, adversarial tool tests | Durable typed conversation state and external multi-turn red team | Good bounded-agent prototype |
| MLE | Grouped temporal splits, leakage/shortcut audits, calibration, perturbation, promotion holds | Simulator-label coupling and no exact-target external cohort | Strong synthetic MLE process, weak outcome evidence |
| Statistics | Bootstrap/paired tests, calibration and subgroup diagnostics | Endpoint validity, homogeneous synthetic samples, no proven superiority | Useful engineering statistics only |
| XAI | Additivity/fidelity checks and suppression of unstable ranked explanations | Retraining rank instability and no comprehension study | Internal diagnostic, not patient explanation evidence |
| Medical safety | Claim boundaries, abstention, urgent routing, post-generation controls, fail-closed transport | Unsafe untouched holdout, zero clinician/genetic-counselor review, no staffed escalation owner | Still blocks real patient use |
| SWE/testing | Modular services, typed frontend, ship tiers, fault tests, release artifacts | Large modules, selected CI subsets, non-hermetic dependency environment | Solid prototype engineering |
| Automation | Leases, heartbeat, retries, dead letters, signed delivery receipts; Compose now launches the leased worker | External channel and human acknowledgement SLO not proven | Ready for synthetic staging drills, not clinical escalation |
| Data engineering | Contracts, hashes, quarantine, lineage, replay drills | Local latest-file operation, no deployed orchestrator/catalog/SLA | Good local evidence, not an operated data platform |
| Infrastructure | Loopback disposable stack, non-root distroless backend, leased worker, Postgres restore, inactive n8n import, and MailHog receipt drill | No disposable cloud deployment, managed restore, secret rotation, or public ingress proof | Strong local synthetic runtime evidence only |
| Security | Hashed demo sessions, browser session-only bearer storage, OIDC/PKCE strict-profile contract, upload quarantine policy, dependency and container scans | Live IdP flow unproven; uploads disabled without an external scanner; container has 15 unfixed high findings | Blocks public network deployment |
| Privacy | Structured redaction and patient-scoped APIs | No complete PHI lifecycle, retention/deletion/export, encryption/key/backup proof | Blocks real data |
| Observability/cost | Request IDs, traces, route latency, token estimates and cache metrics | Process-local telemetry and no provider/billing reconciliation | Useful local diagnostics |
| Deployment | Health/readiness plus dependency-import probes, synthetic-only API boundary, runtime recovery drills, and release warnings | Public image security, managed-service DR, real SLOs, external owners, and institutional controls | Loopback synthetic staging verified; patient deployment not close |

## Implemented in This Pass

### SAF-001: fail-closed RAG release boundary

- A valid typed evidence envelope is now mandatory for evidence-dependent
  answers.
- Only `ALLOW` releases an evidence-dependent reply.
- Missing, malformed, unknown, partial, failed, or stale validation state
  becomes a bounded abstention or block.
- Cache hits require current envelope, policy, safety, validator, KB fingerprint,
  TTL, and response digest checks.
- SSE answer text remains buffered until final authorization.
- JSON, SSE, nested support-chat, cache, live-agent, and evaluation paths are
  covered by the same transport assertion.

### Stop-the-line release evidence

`scripts/run_fail_closed_rag_assurance.py` executes the actual fault-injection
suite and writes
`Data/evals/safety/latest_fail_closed_rag_assurance.json`. The artifact includes
test counts, policy versions, protected response paths, source hashes, and
nonclinical claim boundaries. It is a required release-gate artifact; a missing,
stale, timed-out, under-sized, or failing suite blocks engineering release.

### AUTO-001: deployed worker mismatch

Both Compose profiles now launch `scripts/run_automation_worker.py`, which uses
database leases, heartbeats, retry delays, and dead-letter state. A regression
test rejects either profile if it returns to `run_task_worker.py`. This closes
the specific deployment-wiring defect; it does not prove external delivery or
human acknowledgement.

### Restricted synthetic staging boundary

- Demo bearer tokens are stored only as SHA-256 digests and raw browser tokens
  use `sessionStorage`, not persistent `localStorage`.
- Staging/production-shaped profiles require OIDC authorization-code flow with
  PKCE configuration. This is a configuration/startup contract; a live identity
  provider login and logout have not been demonstrated.
- Mutating API requests in synthetic-only runtime require an explicit synthetic
  data-classification header and synthetic patient namespace.
- Uploads are disabled by default in strict profiles. When enabled, strict
  base64 decoding, magic-byte/type alignment, quarantine-first promotion, and
  an external scanner are mandatory; scanner errors fail closed.
- `latest_restricted_synthetic_staging_assurance.json` is a required hard
  blocker in the engineering release decision surface.

### Executed disposable runtime drill

The loopback Compose profile was built and exercised with seven services:
backend, frontend, leased worker, Postgres, Redis, n8n, and MailHog. The current
runtime artifact records all services running, backend dependency imports
working, all network probes passing, a Postgres dump/restore drill, an inactive
n8n workflow import, and a MailHog-only synthetic delivery receipt. It also
records `patient_data_processed=false` and `real_external_delivery=false`.

The backend image now uses a multi-stage non-root distroless runtime with a
shell-free Python entrypoint. A runtime import probe was added after the drill
found that a Compose `PYTHONPATH` override could make the shallow health route
green while the worker failed to import SQLAlchemy.

### Container and dependency evidence

- Python and npm dependency scans currently report zero known high/critical
  dependency findings; npm audit reports zero vulnerabilities.
- Trivy scanned the exact current backend image. Moving from Debian 12/Python
  3.11 distroless to Debian 13/Python 3.13 reduced severe findings from 2
  critical plus 40 high to 0 critical plus 15 high. None of the remaining 15
  currently has a published fixed version in the scanner feed.
- The canonical decision is `BLOCK_PUBLIC_DEPLOYMENT`. The negative result is a
  visible release warning and is linked into the software supply-chain evidence.
- The scan is point-in-time engineering evidence, not a penetration test or
  security certification.

## Current Gate State

At the first post-change check, the new fail-closed artifact passed while the
canonical release gate correctly remained red because four required artifacts
were older than their 30-day freshness policy. The ship runner was then changed
to regenerate those artifacts with their canonical runners before evaluating
the release gate:

- `latest_safety_benchmark.json`
- `latest_adversarial_eval.json`
- `latest_rag_benchmark.json`
- `latest_rag_intent_aware_eval.json`

The previous full ship workflow passed all 70 selected steps. The current fast
ship tier also passes: 80 breast-monitoring tests, 27 progressive-loading and
notification tests, 103 cloud/data/vector/security tests, 166 assurance/XAI/
automation/safety tests, 57 frontend tests, lint, production build, 68
fail-closed RAG tests, and 25 restricted-staging tests. The release gate reports
231 artifacts, 29 decision artifacts, zero hard failures, and one appendix
warning. The compact decision surface returns `PROCEED_WITH_WARNINGS` with four
visible warnings: frozen adversarial v7, synthetic perturbation stress, XAI
reliability, and container security. The appendix warning is non-zero unsafe
leakage in an internal adversarial-generalization evaluation; it has not been
waived. The repository secret scan reports zero findings.

## Priority Roadmap

### P0: required before any public or managed synthetic staging

1. Track the 15 currently-unfixed high image findings against refreshed Debian
   and Trivy feeds, rebuild/rescan the exact digest, and keep public deployment
   blocked until the explicit image policy is satisfied.
2. Complete a real browser OIDC authorization-code/PKCE login, refresh, logout,
   revocation, and role-mapping drill against a disposable identity tenant.
3. Keep uploads disabled until a real external malware scanner, encrypted object
   store, deletion/retention policy, and cross-patient authorization drill exist.
4. Add multi-worker kill/recover, duplicate-delivery, lease-expiry, Redis outage,
   and idempotency tests to the executable Compose runtime drill.
5. Add rollback, dependency-outage, secret-rotation, and encrypted backup/restore
   evidence in a disposable managed environment.
6. Add an immutable image digest/signature, generated image SBOM, provenance,
   and admission policy rather than deploying a mutable `latest` tag.

### P1: strongest internal evidence improvements

1. Replace high-risk lexical claim acceptance with structured claim
   decomposition plus entailment/contradiction; unavailable semantic validation
   must abstain.
2. Complete a no-read externally authored RAG and adversarial holdout. Internal
   expansion does not substitute for this.
3. Keep only RAG stages that improve a frozen quality-governance-latency frontier;
   preserve the honest result that raw retrieval superiority over BM25 is not
   proven.
4. Make the proxy-removed synthetic feature policy the only promotion-eligible
   trainer and run repeated cross-generator, label-sensitivity, and missingness
   stress tests.
5. Keep patient-facing ML output limited to data availability and review status;
   retain probabilities and unstable XAI in admin/research surfaces.
6. Create immutable data run IDs, schema-contract versioning, idempotent replay,
   and a bad-batch quarantine/recovery demonstration.

### P2: evidence that requires people or institutions

1. Oncology nurse/clinician review of urgent wording, refusal wording, and
   escalation categories.
2. Genetic-counselor review of VUS/genetic-risk handling.
3. Human-factors comprehension testing for overtrust and misunderstanding.
4. Privacy/security review before accepting any real health information.
5. Exact-target external data, governance agreements, and appropriate ethics
   oversight before clinical-performance claims.

## Honest Readiness Summary

- **Local synthetic demo:** deployable for portfolio demonstration; the fresh
  canonical ship run passes.
- **Restricted loopback synthetic staging:** executable engineering drill passes;
  it is suitable for local portfolio/reviewer use with synthetic data only.
- **Public or managed synthetic staging:** still blocked by container findings,
  unexecuted live OIDC, disabled uploads, secret/DR gaps, and absent cloud runtime
  proof.
- **Patient-facing healthcare deployment:** not close and must remain blocked.

The most credible portfolio story is not that NLCare is production healthcare
software. It is that a confirmed fail-open defect and a deployed-worker mismatch
were found through audit, fixed at the correct architectural boundaries, backed
by fault injection, and converted into release-blocking evidence without hiding
the remaining clinical, privacy, and security gaps.
