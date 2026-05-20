# Model and KB Version Rollback Notes

## Model Artifacts

1. Identify the last green model artifact and metadata sidecar.
2. Confirm feature-set version and training-data fingerprint.
3. Restore both model and metadata; do not mix versions.
4. Rerun leakage, calibration, counterfactual stability, and release gate.

## Knowledge Base

1. Identify the last green KB source manifest and index fingerprint.
2. Restore raw curated KB files and rebuilt index together.
3. Rerun ingestion, source governance, retrieval ablation, live RAG, and claim
   citation eval.

## Boundary

Rollback readiness is engineering discipline. It is not clinical validation or
production healthcare disaster recovery.
