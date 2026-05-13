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
- Temporal generalization split: train on earlier synthetic patient timelines, evaluate on later timelines
- High-noise stress test: missing CBC values, lab jitter, WBC unit-entry error, site batch effects, contradictory symptom records
- Calibration comparison: raw probabilities versus isotonic, Platt, and temperature scaling
- Synthetic realism audit: lab threshold coverage and sim-to-real distribution checks against external MRI-feature baselines

Evidence: [backend/services/complete_synthetic_training.py](backend/services/complete_synthetic_training.py), [backend/services/admin_analytics.py](backend/services/admin_analytics.py), [backend/services/evaluation_reports.py](backend/services/evaluation_reports.py), [backend/services/temporal_eval.py](backend/services/temporal_eval.py), [backend/services/noise_eval.py](backend/services/noise_eval.py), [backend/services/calibration_eval.py](backend/services/calibration_eval.py)

Run:
```
python scripts/run_training_pipeline.py --skip-training --model-version synthetic-v1
python scripts/generate_eval_report.py
python scripts/run_mle_checks.py
python scripts/run_temporal_eval.py
python scripts/run_noise_eval.py
python scripts/run_synthetic_realism_report.py
```

Note: `--skip-training` expects existing artifacts under `Data/complete_synthetic_training`. Omit it to generate fresh artifacts.

Current local robustness reports:
- Temporal eval: `Data/mle_monitoring/temporal_eval_report.json`
- Noise eval: `Data/mle_monitoring/noise_eval_report.json`
- Calibration eval: `Data/mle_monitoring/calibration_eval_report.json`
- Synthetic realism: `Data/mle_monitoring/synthetic_realism_report.json`

Claim boundary: these reports are synthetic engineering stress tests. They do not establish real-world clinical safety or effectiveness.

## Workflow evaluation
- Clinician approve, edit, reject, and follow-up rates
- Explanation quality score
- Model usefulness score
- Patient feedback rating
- App event error rate
- Prediction audit count
- Cache hit rate and latency
- Summary quality proxy report using rubric + clinician reviews

Evidence: [backend/services/clinician_feedback.py](backend/services/clinician_feedback.py), [backend/services/agent_feedback.py](backend/services/agent_feedback.py), [backend/services/app_logging.py](backend/services/app_logging.py), [backend/services/summary_quality_eval.py](backend/services/summary_quality_eval.py)

Run:
```
python scripts/run_summary_quality_eval.py
```

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
