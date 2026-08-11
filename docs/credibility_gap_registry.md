# Credibility Gap Registry

This registry prevents green internal tests from being mistaken for external or clinical evidence.

- Open or external gaps: `10`
- Cannot be self-certified: `4`

## Gaps

### provider_token_usage_coverage

- Domain: `AIE/observability`
- Status: `open`
- Severity: `medium`
- Owner: `AI platform owner`
- Completion: Capture provider-reported usage on at least 80% of 30+ controlled representative synthetic requests; staged traffic remains separate.
- Until closed: Provider usage is measured on a controlled synthetic suite only; historical rows remain partly estimated and totals are not audited billing truth.

### tail_latency_evidence

- Domain: `SWE/infra`
- Status: `complete_internal`
- Severity: `high`
- Owner: `SWE/infra owner`
- Completion: Collect 100+ representative requests and pass route-specific p95 budgets without hiding cold starts; reconcile the local probe against staged cloud traffic.
- Until closed: Current p95 is internal and sample-dependent, not a production SLO.

### fine_tune_runtime_and_candidate

- Domain: `MLE/fine-tuning`
- Status: `open`
- Severity: `high`
- Owner: `MLE owner`
- Completion: Pass pinned runtime, lineage, memorization, per-behavior, paired-statistical, safety, and output-length gates.
- Until closed: Fine-tuning is scaffolded only; no adapter improvement is proven.

### fine_tune_semantic_contamination

- Domain: `MLE/evaluation`
- Status: `open`
- Severity: `medium`
- Owner: `MLE evaluation owner`
- Completion: Run semantic/paraphrase contamination detection with reviewer adjudication of flagged pairs.
- Until closed: TF-IDF is a lexical-semantic proxy; even a completed review does not prove semantic independence.

### rag_improvement_over_bm25

- Domain: `AIE/RAG`
- Status: `open`
- Severity: `high`
- Owner: `RAG evaluation owner`
- Completion: Demonstrate a predeclared improvement on an independent no-read holdout, or retain governance-first positioning.
- Until closed: The complex stack is governance-oriented; raw retrieval superiority over BM25 is not proven.

### frozen_adversarial_generalization

- Domain: `AIE/safety`
- Status: `open`
- Severity: `high`
- Owner: `AI safety owner`
- Completion: Meet predeclared frozen-bank thresholds without using the bank for tuning and preserve safe-negative performance.
- Until closed: Safety is not solved; frozen adversarial weaknesses remain visible.

### independent_clean_clone_reproduction

- Domain: `SWE/reproducibility`
- Status: `blocked_external`
- Severity: `high`
- Owner: `Independent peer engineer`
- Completion: A reviewer with no project involvement reproduces setup, tests, artifacts, and demo from a clean clone.
- Until closed: The owner has internal reproducibility evidence, not independent reproduction.

### external_no_read_evaluation

- Domain: `Evaluation governance`
- Status: `blocked_external`
- Severity: `high`
- Owner: `External evaluation author`
- Completion: An eligible external author completes the no-read RAG and adversarial protocols with attestation.
- Until closed: Prepared external evaluation is not completed external evidence.

### clinician_and_genetics_review

- Domain: `Medical governance`
- Status: `blocked_external`
- Severity: `critical`
- Owner: `External clinician and genetic counselor`
- Completion: Qualified reviewers complete dated, case-linked review logs; this still does not equal clinical approval.
- Until closed: Medical wording and boundaries are unreviewed by clinicians.

### live_cloud_and_delivery_evidence

- Domain: `Infra/automation`
- Status: `open`
- Severity: `high`
- Owner: `Infra/automation owner`
- Completion: Run staged live delivery, retry, duplicate suppression, restore, failover, load, and cost reconciliation drills.
- Until closed: Local/synthetic automation readiness is not live cloud reliability.

### real_data_irb_clinical_validation

- Domain: `Clinical evidence`
- Status: `blocked_institutional`
- Severity: `critical`
- Owner: `Clinical institution`
- Completion: Requires institutionally governed real data, ethics/IRB review, clinical protocol, and qualified oversight.
- Until closed: No clinical readiness, real-world safety, patient benefit, or healthcare-production claim is allowed.
