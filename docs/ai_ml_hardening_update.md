# AI/ML Hardening Update

This update moves MedicalAgent closer to a senior-level applied AI engineering
portfolio project while keeping the system account-free by default.

## Added

- GitHub Actions quality gate for backend compilation, focused tests, frontend
  TypeScript build, RAG gold-set eval, safety/RAG/drift evals, MLE readiness,
  API smoke checks, and Docker image build.
- Docker Compose stack with Postgres, Redis, FastAPI backend, DB-backed task
  worker, and Vite React frontend.
- DB-backed async task worker for slow operations such as RAG indexing,
  safety/RAG evals, drift reports, MLE readiness, public imaging reports, and
  ultrasound/CT imaging baselines.
- Fast/offline and slower/live-agent safety/RAG evaluation modes exposed through
  admin API and the Safety Center UI.
- Hand-labeled `evals/rag_gold_cases.json` benchmark plus
  `scripts/evaluate_rag_gold_set.py`.
- Optional cross-encoder reranker for RAG candidates. It is disabled by default
  and enabled with `RAG_ENABLE_CROSS_ENCODER=true`.
- Deterministic post-generation answer verifier for obvious diagnostic,
  treatment-directive, citation, and high-risk escalation failures.
- Optional MLflow mirroring for the local experiment registry via
  `MLFLOW_ENABLED=true`.
- BUSI ultrasound transfer-learning and segmentation baselines, plus admin API
  and dashboard panels. These remain explicitly non-diagnostic engineering
  baselines.

## Current Interpretation

The engineering system is strong for a portfolio PoC: it has safety-first agent
routing, dense/sparse RAG, trace logging, MLE gates, eval artifacts, and visible
admin observability. The main remaining non-technical ceiling is clinical
validity: most end-to-end ML metrics are still synthetic-data results, and the
realism gate correctly stays cautious.

## Next Best Steps

1. Tune synthetic data realism against public dataset distributions and rerun
   the realism gate.
2. Build a small U-Net/transfer segmentation experiment after the classical BUSI
   baseline, keeping the current classical baseline as the floor.
3. Run the live-agent safety/RAG suites after major agent changes, while keeping
   offline deterministic suites as CI gates.
4. Add a Celery/RQ adapter only if the DB-backed worker becomes insufficient.
5. Add a real external validation narrative around the domain gap instead of
   chasing synthetic AUROC.
