# Evaluation & Safety

This is the single landing page for everything evaluation- and safety-related in MedicalAgent. It points to the artifacts, the commands that regenerate them, the tests that pin the contracts, and the known limitations and gaps.

Cross-references:

- [MODEL_CARD.md](../MODEL_CARD.md) — model purpose, metrics, calibration, limitations.
- [DATA_CARD.md](../DATA_CARD.md) — synthetic data assumptions, lineage, leakage prevention.
- [SAFETY_CARD.md](../SAFETY_CARD.md) — non-diagnostic boundaries, refusal rules, escalation logic.
- [PHI_PRIVACY_LIMITATIONS.md](PHI_PRIVACY_LIMITATIONS.md) — what this project is **not** approved for, and what real PHI handling would require.
- [AI_ML_ARCHITECTURE.md](AI_ML_ARCHITECTURE.md) — pipeline diagram and module map.

## Safety & Evaluation Center (dashboard)

Surfaced under the admin dashboard → **Safety & Eval Center** tab, backed by `GET /admin/safety-center`.

| Suite | Artifact | Source |
|---|---|---|
| Safety red-team | `Data/evals/safety/latest_safety_red_team.json` (+ `.csv`) | [`backend/services/safety_red_team.py`](../backend/services/safety_red_team.py) |
| RAG evaluation | `Data/evals/rag/latest_rag_eval.json` | [`backend/services/rag_eval_suite.py`](../backend/services/rag_eval_suite.py) |
| Drift / data quality | `Data/evals/drift/latest_drift_report.json` | [`backend/services/drift_monitoring.py`](../backend/services/drift_monitoring.py) |
| Failure case gallery | `Data/reports/failure_case_gallery.json` | [`backend/services/failure_case_gallery.py`](../backend/services/failure_case_gallery.py) |
| Calibration | `Data/complete_synthetic_training/complete_synthetic_model_metrics.json` | training pipeline |
| Clinician feedback | live DB (`clinical_summary_reviews` table) | [`backend/services/clinician_feedback.py`](../backend/services/clinician_feedback.py) |

Each artifact carries `schema_version`, `generated_at`, a `summary`, per-case detail, and a `limitations` array. Missing artifacts return `{ "status": "not_generated" }` so the dashboard degrades gracefully.

## Commands

### Regenerate Safety & Evaluation Center artifacts

```
python -m scripts.run_safety_eval_center                  # safety + rag + drift
python -m scripts.run_safety_eval_center --only safety    # restrict to one suite
python -m scripts.run_safety_eval_center --only rag
python -m scripts.run_safety_eval_center --only drift
python -m scripts.run_safety_eval_center --print-summary  # dump a JSON summary
```

### Benchmark ladder (portfolio report)

```
python scripts/run_safety_benchmark.py
python scripts/run_rag_benchmark.py
python scripts/run_adversarial_benchmark.py
python scripts/run_model_benchmark.py
python scripts/run_realism_checks.py
python scripts/run_clinician_summary_benchmark.py
python scripts/generate_benchmark_report.py
```

### Demo bootstrap (seed + evals together)

```
python -m scripts.seed_demo_and_evals
```

### Older evaluation scripts (still wired into CI)

```
python scripts/evaluate_agent_rag.py        # agent regression suite
python scripts/run_mle_checks.py            # MLE readiness gate
python scripts/run_noise_eval.py            # noise robustness
python scripts/run_temporal_eval.py         # temporal split / cycle split eval
python scripts/run_summary_quality_eval.py  # summary quality
python scripts/run_quality_gate.py          # local quality gate (frontend + tests)
```

### Tests

```
python -m pytest tests/test_safety_eval_center.py
python -m pytest tests/test_clinician_feedback.py
python -m pytest tests/test_constants_sync.py
python -m pytest tests/test_timeline_uncertainty.py
python -m pytest tests/test_access_control.py
python -m pytest tests/test_public_imaging_services.py
python -m unittest tests/test_breast_monitoring.py   # this is what CI runs
```

### Frontend typecheck

```
cd frontend-react && npx tsc -b --noEmit
```

## Safety design — short version

- **Deterministic guardrails first.** Treatment, medication, and diagnosis refusal routes are decided by pattern + rule, not by LLM judgment. See [SAFETY_CARD.md](../SAFETY_CARD.md).
- **Uncertainty on every AI/model output.** `risk_engine.py` emits `confidence_level`, `uncertainty_reason`, `missing_data_indicators`, and `clinician_review_required` on every risk flag. The patient timeline forwards these via [`backend/processing/timeline.py`](../backend/processing/timeline.py) so the frontend `AIGeneratedLabel` can always render them.
- **Wire-contract test.** [`tests/test_constants_sync.py`](../tests/test_constants_sync.py) parses `frontend-react/src/lib/constants.ts` and asserts the enum arrays match `backend/services/review_constants.py`. This stops the FE and BE drifting on review decisions, risk levels, confidence levels, reason categories, or artifact statuses.

## Known evaluation gaps (honest)

| Gap | What it means | Why it's still listed |
|---|---|---|
| `tests/test_breast_monitoring.py::test_agent_regression_suite_tracks_guardrails_and_sources` is **failing in HEAD** (`attack_block_rate` 0.625 vs expected 1.0) | The agent regression suite's attack-block computation does not currently match the test's expectation. | Failure pre-dates the current upgrade work; neither `agent_regression_eval.py` nor the test file have been changed here. The safety red-team suite (`tests/test_safety_eval_center.py`) covers the same behavior under a different metric — that suite passes 9/9. Tracking this gap publicly is more honest than silently disabling the test. |
| Grounding / hallucination metrics in RAG eval are **heuristic proxies**, not labeled-gold scores. | Numbers move with retrieval quality but they're not human-rated. | Acceptable for engineering monitoring; clearly labeled in `Data/evals/rag/latest_rag_eval.json`'s `limitations` block. |
| BreastDCEDL real-data baseline AUROC ≈ **0.637**. | Synthetic-trained models do not transfer to real MRI-derived signals out of the box. | Recorded explicitly in [MODEL_CARD.md](../MODEL_CARD.md) so portfolio readers see it without scrolling past the synthetic numbers. |
| No external validation beyond BreastDCEDL/I-SPY1. | One external direction is not a clinical validation story. | Honest framing throughout: this is a synthetic-data engineering portfolio, not a clinical-grade system. |
| No PHI / regulatory controls. | Real PHI handling would need DPAs, SOC2-style controls, model-versioning audit, etc. | Out of scope for a portfolio PoC. Stated up-front in README and SAFETY_CARD. |

## Where to add new evaluation cases

| Test type | Add to | Run command |
|---|---|---|
| Safety red-team case | `evals/safety_red_team_cases.json` | `python -m scripts.run_safety_eval_center --only safety` |
| RAG eval case | `evals/rag_eval_cases.json` | `python -m scripts.run_safety_eval_center --only rag` |
| Failure case (gallery) | `Data/reports/failure_case_gallery.json` | Reload dashboard |
| Backend unit test | `tests/test_*.py` | `python -m pytest tests/test_<name>.py` |
| Wire-contract enum | `backend/services/review_constants.py` **and** `frontend-react/src/lib/constants.ts` | `python -m pytest tests/test_constants_sync.py` |

The wire-contract test is deliberately the gating mechanism: changing one side without the other will fail CI before code merges.
