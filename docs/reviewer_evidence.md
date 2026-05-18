# Reviewer Evidence Map

OncoTrack is a safety-first engineering prototype. This file maps each major
claim to the command or artifact a reviewer can inspect. These checks are
engineering evidence only; they do not establish clinical validity.

## Stability Commands

| Claim | Evidence |
|---|---|
| Backend chat/RAG/safety integration is currently stable | `RAG_FORCE_SPARSE=true python -m pytest tests/test_breast_monitoring.py -q` |
| Frontend unit behavior is currently stable | `cd frontend-react && npm run test` |
| Frontend smoke flows work in a browser | `cd frontend-react && npm run test:e2e -- tests/e2e/smoke.spec.ts` |
| Frontend lint/build are clean | `cd frontend-react && npm run lint && npm run build` |
| Release artifacts are fresh and above configured thresholds | `python scripts/run_release_gate.py` |
| Full local ship gate works cross-platform | `python scripts/ship.py` |

## Governance Artifacts

| Area | Artifact |
|---|---|
| Leakage audit | `Data/evals/models/latest_leakage_audit.json` |
| Evidence-aware abstention | `Data/evals/models/latest_evidence_abstention_eval.json` |
| Modality robustness | `Data/evals/models/latest_modality_robustness_comparison.json` |
| Counterfactual stability | `Data/evals/models/latest_counterfactual_stability.json` |
| Per-head calibration | `Data/evals/models/latest_per_head_calibration.json` |
| Shortcut audit | `Data/evals/models/latest_shortcut_audit.json` |
| RAG safety/quality | `Data/evals/rag/latest_rag_benchmark.json` |
| RAG intent-aware behavior | `Data/evals/rag/latest_rag_intent_aware_eval.json` |
| RAG source governance | `Data/evals/rag/latest_kb_source_governance.json` |
| Medical claim boundary | `Data/evals/safety/latest_medical_claim_boundary_eval.json` |
| Failure-mode registry | `Data/evals/safety/latest_failure_mode_registry.json` |
| Medical advisor review packet | `Data/evals/medical/latest_medical_advisor_review_packet.json` |

## Prediction Traceability

Live evidence-aware predictions persist trace rows with model version,
feature-set version, threshold/calibration config, evidence sufficiency,
modalities present/missing, request id, actor role, and timeline snapshot hash.
Reviewer path:

1. Log in as clinician.
2. Open a patient detail view.
3. Inspect the prediction trace panel.
4. Compare the displayed prediction envelope against the persisted trace fields.

## What Must Not Be Claimed

- The system is clinically validated.
- The model predicts real patient treatment success.
- The RAG layer gives medical advice.
- The agent diagnoses cancer, recurrence, or toxicity.
- Genetic counseling readiness interprets inherited risk.
- Biomarker or tumor-marker records determine treatment.
- Supplement information means a product is safe or effective.
- Synthetic benchmark performance transfers to real patients.

## Honest Positioning

OncoTrack demonstrates controllable engineering discipline before clinical
validation: leakage prevention, abstention, traceability, source-governed RAG,
medical claim boundaries, release gates, and clinician-review workflows. It is
not a clinical product and is not approved for real patient care.
