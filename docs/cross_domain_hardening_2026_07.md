# Cross-Domain Hardening Pass (July 2026)

NLCare remains a synthetic-only, non-diagnostic engineering prototype. This
pass improves software controls and evaluation visibility. It does not add
clinical authority, clinical validation, patient-benefit evidence, clinician
approval, or production healthcare readiness.

## Implemented Controls

### AIE and bounded agent workflows

- Stateful write confirmations are bound to a patient scope, planned route,
  write tool, normalized action payload, expiry, and consumed/revoked state.
- Payload substitution, stale confirmation, replay, and cross-patient scope
  mismatches fail closed.
- Trusted-looking memory entries with a different patient scope are treated as
  untrusted and block authority-bearing actions.
- The live confirmed-write path remains the only persistence path; the
  orchestrator continues to be a bounded engineering scaffold.

### Automation

- High-risk review notifications retain a local outbox as the source of truth.
- Delivery receipts follow a monotonic state machine and reject backward,
  stale, or implausibly future-dated callbacks.
- Channel delivery remains explicitly distinct from human acknowledgement,
  clinical review, emergency coverage, and clinical action.

### Synthetic MLE and statistics

- The row-level statistical audit now reports decision-threshold sensitivity,
  Wilson intervals, probability-jitter stability, and synthetic prevalence
  reweighting.
- These tests expose brittleness in exported synthetic predictions. They do not
  simulate realistic covariate shift, retrain under perturbation, or establish
  transportability to real patients.
- Promotion remains `hold_synthetic_only`.

### XAI and medical human factors

- The patient XAI dossier inspects the implemented KPI source instead of only
  checking a written specification.
- Patient labels use synthetic comparison-group wording and demonstration
  reference bands instead of clinical-sounding response or toxicity headlines.
- Patient explanations retain meaning, calculation, missingness, safe next
  steps, and explicit non-authority boundaries.

### Data engineering

- The existing non-patient medallion pipeline retains content-addressed bronze,
  validated silver, provider-neutral gold, quarantine, lineage, schema
  migration, tombstones, and deterministic backfills.
- A 100x synthetic identifier replay now checks partition determinism and
  uniqueness without claiming cloud throughput or patient-data readiness.

### Infrastructure and deployment

- The Azure Bicep reference blocks the optional application unless its image is
  digest-pinned, its database secret is referenced, private networking is
  enabled, and public network access is disabled.
- Optional Search, Service Bus, and PostgreSQL resources are guarded by private
  networking and public-network-disabled conditions.
- These are compiled reference controls only. No Azure deployment, authenticated
  what-if, private-connectivity exercise, or cloud recovery test was completed.

### SWE and release discipline

- Every `scripts/ship.py` step now has a bounded timeout and duration record.
- Ship runs emit `Data/evals/ops/latest_ship_run.json` on pass, timeout, or
  non-zero exit.
- A passing ship manifest is local engineering evidence only.

## Deliberately Not Promoted

- Iterative RAG remains eval-only because the internal result showed no
  insufficiency reduction.
- The citation pruner remains experimental because it reduced citation
  precision.
- The cross-encoder reranker remains optional because retrieval improvement is
  not proven.
- The full source-governed RAG stack remains valuable for source-policy
  correctness, but raw Recall@10 superiority over BM25 is not proven.
- Synthetic response and toxicity outputs remain monitor-only or
  review-hint-only.

## Remaining External Blockers

- No-read external RAG holdout completion.
- External-author adversarial cases.
- Clinician or oncology-nurse safety wording review.
- Genetic-counselor VUS review.
- Real dataset access with a task-aligned target and appropriate governance.
- Authenticated cloud deployment, load, restore, and security evidence.

None of those blockers can be replaced by adding more internal artifacts.
