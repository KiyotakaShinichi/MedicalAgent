# ML Lifecycle

MedicalAgent uses a local, account-free MLE workflow designed for engineering practice and reproducibility, not clinical deployment.

## Lifecycle stages
1. Synthetic longitudinal dataset generation
   - Evidence: [backend/services/synthetic_journey.py](backend/services/synthetic_journey.py)
2. Feature-store materialization
   - Evidence: [backend/services/feature_store.py](backend/services/feature_store.py)
3. Training runs and metrics
   - Evidence: [backend/services/complete_synthetic_training.py](backend/services/complete_synthetic_training.py)
4. Artifact generation and evaluation reports
   - Evidence: [backend/services/evaluation_reports.py](backend/services/evaluation_reports.py)
5. Readiness gates and data contracts
   - Evidence: [backend/services/mle_readiness.py](backend/services/mle_readiness.py)
6. Registry metadata and promotion or rollback
   - Evidence: [backend/services/model_artifacts.py](backend/services/model_artifacts.py)
7. Audit logging and monitoring
   - Evidence: [backend/services/app_logging.py](backend/services/app_logging.py), [backend/models.py](backend/models.py)

## Readiness gate configuration
- Gate definitions are documented in [config/readiness_gates.yaml](config/readiness_gates.yaml).
- Source of truth remains [backend/services/mle_readiness.py](backend/services/mle_readiness.py).

## Commands
```
python scripts/run_training_pipeline.py
python scripts/materialize_feature_store.py
python scripts/run_mle_checks.py
```

## Non-diagnostic boundary
ML outputs are monitoring signals and risk flags for clinician review, not diagnoses or treatment recommendations.
