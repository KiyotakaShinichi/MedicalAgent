# Evaluation

MedicalAgent evaluation is designed to demonstrate disciplined engineering practice, not clinical validation. The evaluation suite includes RAG, safety, ML, and workflow measures.

## RAG evaluation
- Retrieval precision@3 proxy
- Groundedness proxy
- Citation presence and missing-citation detection
- Unsupported claim and refusal behavior
- Latency, token estimate, and cache-path tracking

Evidence: [backend/services/agent_regression_eval.py](backend/services/agent_regression_eval.py), [backend/services/rag_analytics.py](backend/services/rag_analytics.py), [evals/README.md](evals/README.md)

Run:
```
python scripts/evaluate_agent_rag.py
python scripts/run_rag_eval.py
```

## Safety evaluation
- Diagnosis-seeking prompts
- Medication-change prompts
- Emergency or urgent symptom prompts
- Prompt injection attempts
- Privacy boundary violations
- Case-level route accuracy and input-guardrail status
- Correct refusal and escalation behavior per safety case

Evidence: [backend/services/security_guardrails.py](backend/services/security_guardrails.py), [backend/services/agent_regression_eval.py](backend/services/agent_regression_eval.py), [evals/README.md](evals/README.md)

Run:
```
python scripts/run_safety_eval.py
```

## ML evaluation
- AUROC and PR-AUC
- Brier score and calibration curves
- Sensitivity and specificity
- False-negative review
- Subgroup checks
- Drift and missingness proxies

Evidence: [backend/services/complete_synthetic_training.py](backend/services/complete_synthetic_training.py), [backend/services/admin_analytics.py](backend/services/admin_analytics.py), [backend/services/evaluation_reports.py](backend/services/evaluation_reports.py)

Run:
```
python scripts/run_training_pipeline.py --skip-training --model-version synthetic-v1
python scripts/generate_eval_report.py
python scripts/run_mle_checks.py
```

Note: `--skip-training` expects existing artifacts under `Data/complete_synthetic_training`. Omit it to generate fresh artifacts.

## Workflow evaluation
- Clinician approve, edit, reject, and follow-up rates
- Explanation quality score
- Model usefulness score
- Patient feedback rating
- App event error rate
- Prediction audit count
- Cache hit rate and latency

Evidence: [backend/services/clinician_feedback.py](backend/services/clinician_feedback.py), [backend/services/agent_feedback.py](backend/services/agent_feedback.py), [backend/services/app_logging.py](backend/services/app_logging.py)

## Verification scripts
- Cache safety policy check:
	```
	python scripts/check_cache_safety_policy.py
	```
- Safety eval catalog runner:
	```
	python scripts/run_safety_eval.py
	```
- Audit logging verification:
	```
	python scripts/verify_audit_logging.py
	```
- Model registry metadata validation:
	```
	python scripts/validate_model_registry_metadata.py
	```

## Evaluation catalog
See [evals/README.md](evals/README.md) for the evaluation catalog and scope.
