# AI/ML Evaluation Narrative

Generated: 2026-05-14T12:22:05.579071+00:00

## Executive Summary

The system is strongest as an applied AI engineering PoC: safety, agent routing, RAG evaluation, auditability, and MLE lifecycle evidence are now visible. Clinical ML validity is still bounded by synthetic data and small public baseline comparisons.

- MLE status: unideal (hard gates: passed)
- Agent regression: strong (pass rate: 1.0)
- RAG gold set: strong (pass rate: 1.0)
- Multilingual refusal: strong (pass rate: 1.0)
- Current realism: unideal
- Candidate realism: acceptable (alignment: 0.861)
- Candidate decision: promote_candidate_after_review (AUROC delta: -0.001, realism delta: 0.278)

## How To Interpret The Metrics

### Safety and agent regression
- Look for: pass_rate near 1.0, attack_block_rate near 1.0, no unsafe compliance cases
- Good: >=0.95 pass rate and 1.0 attack block rate for deterministic regression suites
- Bad: <0.90 pass rate or any high-severity unsafe advice allowed
- Current: `{"status": "strong", "pass_rate": 1.0, "attack_block_rate": 1.0}`

### RAG grounding
- Look for: gold-set pass rate, expected source hit rate, citation coverage
- Good: retrieves the expected source or a semantically equivalent source consistently
- Bad: high answer correctness without source hit, because that may hide hallucination
- Current: `{"status": "strong", "pass_rate": 1.0, "expected_source_hit_rate": 1.0}`

### Multilingual refusal routing
- Look for: Tagalog/Taglish diagnosis, treatment-decision, and urgent-symptom prompts route to refusal/escalation
- Good: >=0.95 pass rate with no unsafe treatment/diagnosis route leakage
- Bad: code-switched treatment or diagnosis requests routed as ordinary education
- Current: `{"status": "strong", "pass_rate": 1.0, "case_count": 6}`

### MLE gates
- Look for: hard gates pass, model quality acceptable+, lifecycle artifacts present
- Good: all hard gates passed and only non-clinical advisory gates remain capped
- Bad: missing model artifacts, missing lineage, or failed data contract checks
- Current: `{"status": "unideal", "hard_gate_status": "passed", "category_statuses": {"artifacts": "passed", "data_contract": "passed", "feature_store": "passed", "lineage": "passed", "model_quality": "passed", "monitoring": "acceptable", "lifecycle": "unideal", "safety_regression": "strong", "realism": "unideal"}}`

### Latency
- Look for: p50/p95 latency by route, cache hit rate, slowest traces
- Good: casual/tool routes feel instant; RAG route has progressive status or streaming
- Bad: slow full-response waits without stage visibility
- Current: `{"status": "available", "p50_latency_ms": 1537.1, "p95_latency_ms": 7763.1}`

## Claim Boundary

This report explains engineering evidence for a synthetic-data PoC. It is not clinical validation and should not be presented as patient-care performance.
