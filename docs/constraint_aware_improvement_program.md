# Constraint-Aware Improvement Program

Constraint-aware engineering plan only. Completion means stronger internal engineering evidence, not clinical validation, clinician approval, real-world patient safety, or production healthcare readiness.

## Canonical Priorities

| Priority | Domain | Evidence state | Next internal action |
| --- | --- | --- | --- |
| P1 | aie | needs_attention | Increase independent route sample counts and preserve BM25 as the explicit simple baseline. |
| P2 | automation | verified_internal_only | Exercise one synthetic test-recipient channel and record receipt latency without claiming clinical coverage. |
| P2 | data_engineering | verified_internal_only | Add partition-scale benchmarks before introducing Databricks or Data Factory. |
| P2 | deployment | verified_internal_only | Add a shared encrypted PKCE transaction store, reviewed token exchange, and provider logout before any live browser login claim. |
| P2 | infrastructure | verified_internal_only | Run an authenticated what-if in a disposable non-patient development subscription. |
| P2 | medical | external_evidence_incomplete | Run non-clinical comprehension and overtrust testing with synthetic scenarios. |
| P2 | mle | verified_internal_only | Keep saturated metrics out of headline surfaces and expand ambiguous-label sensitivity analyses. |
| P2 | swe | verified_internal_only | Reduce stale artifact volume and add a Linux reproducibility lane with SBOM and container scan evidence. |

## Domain Acceptance Criteria

### Aie

Prefer the simplest governed retrieval and routing policy that survives frozen evaluation.

- critical safety regression remains passing
- source-tier correctness remains 1.0 on patient-facing routes
- raw retrieval superiority is claimed only when frozen comparison proves it
- stage latency and marginal quality are reported together
- External blocker: No-read RAG holdout and external-authored adversarial cases remain incomplete.

### Mle

Make synthetic ML statistically defensible as engineering evidence while preventing clinical interpretation.

- patient overlap and temporal leakage remain zero
- simple baselines are always reported beside nonlinear models
- selective risk versus coverage is reported for every prediction head
- synthetic confidence is never described as patient outcome probability
- External blocker: Simulator-built labels cannot establish real-patient transfer or clinical calibration.

### Swe

Make the system reproducible, reviewable, secure by default, and boring to operate in controlled demos.

- full ship workflow passes from a clean checkout
- hard blockers cannot be diluted by informational artifacts
- canonical evidence is refreshed after its dependencies
- dependency and container scans expose unavailable tools as gaps
- External blocker: No production traffic, penetration test, SOC 2, HITRUST, or healthcare security review exists.

### Data Engineering

Operate a lineage-complete, idempotent non-patient knowledge pipeline with explicit recovery behavior.

- all gold records retain governance metadata
- duplicate identifiers and hard data-quality failures remain zero
- replay produces the same KB fingerprint
- no patient data or raw chat enters managed-vector payloads
- External blocker: No cloud lakehouse execution or real-patient data-quality evidence exists.

### Infrastructure

Keep infrastructure reproducible, private by default, cost-bounded, and disposable.

- Bicep compilation has zero diagnostics
- public network and cost-bearing resources remain opt-in
- no secrets are committed
- deployment evidence distinguishes compile, what-if, deploy, restore, and teardown
- External blocker: No cloud deployment, private-connectivity exercise, restore drill, or cost measurement exists.

### Medical

Preserve clinician authority and make uncertainty, missingness, and non-diagnostic boundaries understandable.

- diagnosis, prognosis, treatment, dosage, and genetic/tumor-marker conclusions remain blocked
- patient explanations include meaning, calculation, missingness, limitation, and safe review action
- delivery receipts are never described as clinician acknowledgement
- all review packets remain labelled unreviewed until actually completed
- External blocker: No clinician, nurse, genetic counselor, pharmacist, or patient review has been completed.

### Automation

Automate redacted engineering workflows with durable delivery semantics and explicit human acknowledgement boundaries.

- duplicate enqueue and replay remain idempotent
- PHI and raw chat are rejected from external payloads
- delivery receipt and human acknowledgement remain separate fields
- patient-facing or clinical action automation stays disabled
- External blocker: No provider reliability, operator rota, emergency coverage, or clinician acknowledgement evidence exists.

### Deployment

Support controlled non-clinical deployment profiles without implying healthcare production readiness.

- demo auth and localhost CORS are disabled in strict profiles
- OIDC issuer, audience, HTTPS JWKS, and exact role mapping fail closed
- PKCE callback transactions are single-use and verifier material is not disclosed
- production healthcare readiness remains false
- External blocker: No live identity provider, managed database recovery, production SLO history, or healthcare deployment review exists.

## What This Cannot Prove

- clinical validation or patient benefit
- real-world safety or medical correctness
- generalisation from synthetic ML to real patients
- managed-cloud reliability without deployment evidence
- human acknowledgement from an automation delivery receipt
- production healthcare readiness
