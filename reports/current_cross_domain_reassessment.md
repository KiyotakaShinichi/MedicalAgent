# NLCare Cross-Domain Engineering Reassessment

Generated: 2026-08-09

## Scope and boundary

NLCare remains a restricted, synthetic-only, non-diagnostic engineering
prototype. This assessment does not establish clinical validation, patient
benefit, clinician approval, IRB approval, regulatory compliance, managed-cloud
resilience, or production healthcare readiness.

## Current verdict

The project demonstrates unusually broad applied AI systems engineering for a
student portfolio, but breadth is no longer its limiting factor. The limiting
factor is independent evidence: frozen adversarial generalization, external RAG
holdouts, clinician review, real-data transfer, and managed deployment evidence.

The compact engineering decision is `PROCEED_WITH_WARNINGS` for restricted
synthetic staging only. Public deployment remains blocked.

| Area | Constraint-aware score | Evidence-based assessment |
|---|---:|---|
| AI engineering | 8.0/10 | Strong routing, fail-closed evidence envelopes, traceability, and measured provider usage. Frozen adversarial v7 remains weak. |
| RAG | 7.0/10 | Strong source governance and refusal correctness; raw Recall@10 superiority over BM25 is not proven. The running lightweight staging image uses sparse fallback, now disclosed by readiness. |
| ML/MLE | 7.0/10 | Strong leakage, calibration, temporal split, perturbation, abstention, and promotion controls. All outcome evidence remains simulator-built and degrades under generator shift. |
| XAI | 8.0/10 | Mechanical fidelity is measured and unstable exact ranks are now hidden. Only stable grouped factors are shown alphabetically. No human comprehension or real-data transfer evidence exists. |
| SWE | 8.5/10 | Modular API/UI, tests, release gates, strict auth/deployment profiles, reproducible index manifests, and fail-closed readiness. Artifact volume and legacy surfaces still add maintenance cost. |
| Data engineering | 7.5/10 | Versioned local pipelines, quality contracts, replay/quarantine drills, and lineage evidence are strong. No managed lakehouse or real external pipeline has been exercised. |
| Automation | 8.0/10 | Leased worker, retry/dead-letter controls, idempotency, signed delivery, and runtime recovery are proven locally. No human acknowledgement or external channel SLA is proven. |
| Infrastructure | 7.5/10 | Non-root distroless image, digest-pinned bases, SBOM, OCI labels, Postgres/Redis staging, and restore drill. Fifteen unfixed high CVEs and no managed-cloud drill remain. |
| Medical governance | 6.5/10 | Conservative boundaries, evidence policies, refusal/escalation, and review packets are substantial. No clinician, nurse, pharmacist, or genetic counselor has reviewed the system. |
| Restricted synthetic deployability | 8.0/10 | The loopback stack is repeatable and observable. It is not a public or clinical deployment posture. |
| Real clinical readiness | 1.5/10 | No real patient data, clinician-reviewed labels, prospective workflow study, IRB, regulatory work, or clinical sign-off. |

Scores are engineering judgments, not benchmark metrics.

## Evidence that improved in this pass

1. Controlled provider telemetry ran 30 synthetic research-grounded queries
   through the real patient-agent pipeline. Provider usage was present on 28/30
   queries (93.33%), totaling 16,910 provider-reported tokens. Response content
   was not retained.
2. Warm latency was p50 1,924.4 ms and p95 4,213.87 ms. The cold first request
   took 263,825.07 ms and remains visible rather than being averaged away.
3. Patient XAI now permits only stable consensus factor groups, suppresses
   low-consensus and near-outcome-proxy factors, hides numeric SHAP values, and
   orders factors alphabetically rather than implying stable importance rank.
4. Container bases are pinned by digest. The exact backend image has OCI build
   labels and a CycloneDX 1.7 SBOM with 99 components.
5. The container scan reports 0 critical, 15 high, and 0 fixable high/critical
   findings. Public deployment remains blocked while any severe finding remains.
6. A live disposable recovery drill expired a worker lease, recovered and
   completed the job, suppressed duplicate replay, restored all 36 PostgreSQL
   tables, and matched the normalized source/restore content SHA-256.
7. Persisted RAG indexes now use deployment-specific paths and sidecar manifests
   containing dependency versions, backend identity, KB fingerprint, and index
   SHA-256. Incompatible or tampered indexes fail closed before deserialization.
8. Production readiness can require dense retrieval. If the runtime only has the
   sparse TF-IDF/BM25 fallback, `/ready` returns not-ready instead of silently
   claiming the promised dense capability.

## Highest-risk evidence gaps

1. Frozen adversarial v7 passes 96/142 (67.61%). Prompt injection is 0/10,
   cross-patient exfiltration 1/10, privacy/PII 2/10, prognosis 5/10, dosage
   6/10, and safe negative controls 25/32. This bank must not be tuned on.
2. On the 74-case internal RAG goldset, the full governed stack has Recall@10
   0.7838 versus BM25 0.8041. Source-tier and refusal correctness are 1.0, but
   citation precision is 0.5243 and unsupported context is 0.1622.
3. The controlled provider run reveals a large direct-script cold start. The
   web service now prewarms retrieval, but the lightweight serving image is a
   sparse fallback unless a dense dependency profile is installed.
4. Synthetic perturbation testing remains `HOLD_SYNTHETIC_ONLY`; the largest
   tested generator-shift AUROC delta is -0.141673 and regression MAE deltas
   reach 11.579305 under severity-dependent missingness.
5. Exact XAI rank remains unstable and no human comprehension test exists.
6. Fine-tuning remains scaffolded only. No adapter, matched candidate output,
   or frozen safety comparison has been produced.
7. No external-author RAG holdout, external adversarial author, or clinical
   reviewer has completed a review.

## Next logical work

### Controllable now

1. Build a separate development-only mutation bank for prompt injection,
   privacy, exfiltration, prognosis, dosage, and safe controls. Tune only on that
   bank, freeze it, then run v7 once. Never copy v7 strings or aliases.
2. Add a serving dependency profile that actually includes the dense encoder,
   build its index in CI, and compare its image size, cold start, CVEs, p95,
   retrieval quality, and cost against the sparse serving profile. Do not make
   dense the default unless the trade is favorable.
3. Resolve or formally risk-accept upstream-unfixed container CVEs with expiry,
   owner, compensating controls, and recurring rescan. Public deployment remains
   blocked until policy is satisfied.
4. Run a matched fine-tuning experiment only after the offline runtime and
   contamination review pass. Promotion must require no safety regression and a
   measurable task-specific gain over the base prompt/RAG baseline.
5. Exercise OIDC/PKCE against a real staging identity provider, including key
   rotation, expiry, logout, role revocation, and cross-tenant denial.
6. Run the managed vector store as a read-only shadow. Compare recall, source
   governance, latency, cost, namespace isolation, and delete propagation before
   considering primary traffic.

### Requires another human or external environment

1. Complete the no-read external RAG holdout and source-filter adjudication.
2. Obtain external adversarial cases from an author who has not read the rules or
   current failure bank.
3. Obtain focused wording and workflow review from an oncology nurse/clinician,
   genetic counselor, and pharmacist. This is review, not approval.
4. Run managed Postgres restore, worker failover, real notification delivery,
   and human acknowledgement drills in a restricted cloud staging environment.

## Access ports

| Service | URL/port |
|---|---|
| React frontend | http://127.0.0.1:5173 |
| FastAPI backend | http://127.0.0.1:8017 |
| FastAPI health | http://127.0.0.1:8017/health |
| FastAPI readiness | http://127.0.0.1:8017/ready |
| n8n | http://127.0.0.1:5678 |
| MailHog UI | http://127.0.0.1:8025 |
| MailHog SMTP | 127.0.0.1:1025 |
| PostgreSQL | 127.0.0.1:55432 |
| Redis | 127.0.0.1:56379 |

All running services are part of a disposable synthetic staging environment.
