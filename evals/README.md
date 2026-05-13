# Evaluation Catalog

This folder describes the evaluation suites that make the project credible as an applied healthcare AI system. The goal is not to prove clinical validity. The goal is to prove disciplined engineering behavior: retrieval quality, refusal accuracy, safety boundaries, summary quality, workflow feedback, and model lifecycle readiness.

## Current Automated Suites

Run all backend tests:

```text
python -m unittest tests.test_breast_monitoring
```

Run patient-agent RAG and safety regression:

```text
python scripts/evaluate_agent_rag.py
```

Validate documented eval catalog coverage:

```text
python scripts/evaluate_eval_catalogs.py
```

Run MLE readiness gates:

```text
python scripts/run_mle_checks.py
```

Generate versioned model evaluation artifacts:

```text
python scripts/run_training_pipeline.py --skip-training --model-version synthetic-v1
```

## Eval Families

### RAG Eval

Tests:

- Retrieval precision@3 proxy.
- Expected source hit rate.
- Citation presence.
- Answer grounding proxy.
- Hallucination-risk proxy.
- Unsupported claim/refusal behavior.
- Latency, token estimate, and cache path.

Implemented now:

- `backend/services/agent_regression_eval.py`
- `backend/services/rag_analytics.py`
- `Data/agent_eval/latest_agent_regression.json`
- `evals/rag_cases.json` and `evals/safety_cases.json` are wired into the regression suite

Future upgrade:

- Add labeled research/guideline source sets.
- Add RAGAS/Langfuse-style faithfulness and context relevance scoring.
- Split source quality by guideline, paper, project KB, and synthetic note.

### Safety Eval

Tests:

- Diagnosis-seeking prompts.
- Medication-change prompts.
- Emergency/urgent symptoms.
- Prompt injection and jailbreaks.
- Cross-patient data exfiltration.
- Encoded/obfuscated payloads.
- Mixed-language privacy attacks.
- Benign self-scoped portal-help false positives.

Implemented now:

- Deterministic multilingual guardrails.
- Optional Groq adjudication after deterministic checks.
- Ollama as local learning/fallback experiment.
- Unit tests for encoded, CJK, Tagalog, Spanish, and self-scoped upload-help cases.
- Expanded red-team cases in [evals/safety_cases.json](evals/safety_cases.json).

### Summary Eval

Target summary content:

- Treatment cycle context.
- CBC/lab trend.
- Symptoms.
- Imaging change.
- Risk flags.
- Clinician action needed.
- Uncertainty and missing data.
- Non-diagnostic disclaimer.

Current proxy:

- Clinician summary review logs track approve/edit/reject, notes, explanation quality, and model usefulness.
- Summary quality eval script computes rubric completeness and proxy safety rates.

Future upgrade:

- Add golden synthetic timeline cases.
- Score summary completeness, factual consistency, missing-critical-info rate, and hallucinated-info rate.

### ML Eval

Metrics:

- AUROC.
- PR-AUC.
- Sensitivity/specificity.
- Confusion matrix.
- Brier score.
- Expected calibration error.
- Bootstrap confidence intervals.
- False-negative review.
- Subgroup performance.
- Drift proxy and missingness checks.

Implemented now:

- Versioned evaluation reports under `Data/model_evaluation_reports/`.
- MLE readiness report under `Data/mle_monitoring/latest_mle_readiness.json`.
- Local feature-store manifest under `Data/feature_store/`.
- Local MLOps run tracking under `Data/mlruns_local/`.

### Workflow Eval

Metrics:

- Clinician approve/edit/reject rate.
- Explanation quality score.
- Model usefulness score.
- Patient feedback rating.
- App event error rate.
- Prediction audit count.
- Cache hit rate and latency.

Implemented now:

- `ClinicalSummaryReview`
- `AgentResponseFeedback`
- `AppEventLog`
- `PredictionAuditLog`
- Admin/MLE dashboard summaries.

## Current Quality Snapshot

This section is only valid after you run the evaluation scripts. Do not quote specific statuses unless they were generated locally.

To generate current statuses:

```text
python scripts/evaluate_agent_rag.py
python scripts/run_mle_checks.py
```

Then review:

```text
Data/agent_eval/latest_agent_regression.json
Data/mle_monitoring/latest_mle_readiness.json
```

The expected posture for a synthetic-data healthcare AI PoC is: demoable and testable with clear limitations, not production or clinical validation.

## Claim Boundary

Use:

> Engineering readiness and supervised PoC demo readiness.

Do not use:

> Clinical validation, diagnosis readiness, production healthcare deployment readiness, or HIPAA-certified system.
