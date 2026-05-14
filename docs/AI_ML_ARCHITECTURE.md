# AI / ML Architecture

This document summarizes the AI/ML/MLE layers of MedicalAgent and how
they connect. It is the engineering counterpart to
[../MODEL_CARD.md](../MODEL_CARD.md), [../DATA_CARD.md](../DATA_CARD.md),
and [../SAFETY_CARD.md](../SAFETY_CARD.md).

## High-level pipeline

```
Patient / clinician message
   │
   ▼
┌──────────────────────────────┐
│ Deterministic guardrails     │  input scope, PII boundary, prompt-injection,
│ (backend/services/agent_rag) │  diagnosis-refusal, treatment-refusal patterns
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Intent router                │  general_support / education / portal_help /
│                              │  safety_boundary / treatment_decision_boundary /
│                              │  security_boundary / urgent_symptom_escalation
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Query rewrite + decompose    │  optional, expands query and subqueries
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Hybrid retrieval             │  Dense (FAISS, sentence-transformer)
│ (rag_vector_index)           │  + sparse (BM25)
│                              │  + RRF fusion / reranker / contextual compression
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Citation-checked generation  │  patient-friendly phrasing layer
│ (local_llm or template)      │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Output guardrails            │  PII / directive / hallucination checks
└──────────────┬───────────────┘
               ▼
        Audit + RAG trace log
```

Implementation: [backend/services/agent_rag.py](../backend/services/agent_rag.py), [backend/services/rag_vector_index.py](../backend/services/rag_vector_index.py), [backend/services/local_llm.py](../backend/services/local_llm.py).

## Deterministic vs. LLM safety

Safety-critical behavior is **never** delegated to LLM judgment alone:

- Medication / treatment / diagnosis refusals → deterministic routing in `agent_rag.py`.
- CBC / symptom escalation → rule thresholds in [backend/processing/risk_engine.py](../backend/processing/risk_engine.py).
- Prompt-injection blocks → pattern-driven input guardrail.

The LLM is used for *grounded explanation* and *patient-friendly phrasing* **after** the deterministic layer decides the route. See [SAFETY_CARD.md](../SAFETY_CARD.md).

## Uncertainty layer

Every risk flag produced by `risk_engine.py` carries an `uncertainty` block:

```json
{
  "confidence_level": "moderate",
  "uncertainty_reason": "Trend-based signal with limited context.",
  "missing_data_indicators": [],
  "clinician_review_required": true
}
```

Downstream surfaces render this through the [`AIGeneratedLabel`](../frontend-react/src/components/ui/AIGeneratedLabel.tsx) component, so confidence, uncertainty wording, and the "Clinician review required" badge always appear together. The TimelinePanel detects AI-flag events via `event.ai_generated` or `type` keywords and pins the label onto them.

## Tabular ML lifecycle

Implementation: [backend/services/complete_synthetic_training.py](../backend/services/complete_synthetic_training.py), [backend/services/model_artifacts.py](../backend/services/model_artifacts.py).

- Training discipline: **patient-level** train / calibrate / locked-holdout splits — never row-level.
- Champion classifier: gradient boosting; calibrated with isotonic regression on a dedicated synthetic calibration split.
- Continuous response regressor: robust median ensemble selected by outlier-aware score (MAE + 0.15 × RMSE).
- Hybrid signal: `0.65 × classifier_p + 0.35 × normalized_regression_score`, with agreement bands.

Tracked metrics in the admin dashboard:

- AUROC, AUPRC, Brier, ECE (pre/post temperature), sensitivity, specificity, precision.
- Decision-curve net benefit and cost-sensitive threshold policies (engineering simulations only).
- Subgroup checks by stage / subtype / age / regimen.
- Locked-holdout manifest hash + patient split hash for reproducibility.

## Safety & Evaluation Center

Surfaced under `/admin/safety-center`, with backing artifacts:

| Suite | Artifact | Service |
|-------|----------|---------|
| Safety red-team | `Data/evals/safety/latest_safety_red_team.json` (+ `.csv`) | [safety_red_team.py](../backend/services/safety_red_team.py) |
| RAG evaluation | `Data/evals/rag/latest_rag_eval.json` | [rag_eval_suite.py](../backend/services/rag_eval_suite.py) |
| Drift / data quality | `Data/evals/drift/latest_drift_report.json` | [drift_monitoring.py](../backend/services/drift_monitoring.py) |
| Failure case gallery | `Data/reports/failure_case_gallery.json` | [failure_case_gallery.py](../backend/services/failure_case_gallery.py) |
| Aggregator | (live) | [safety_eval_center.py](../backend/services/safety_eval_center.py) |

Each artifact carries `schema_version`, `generated_at`, a `summary`, per-case detail, and a `limitations` array. Missing artifacts return `{ "status": "not_generated" }` so the dashboard degrades gracefully.

## Clinician feedback loop

- Schema: [`ClinicalSummaryReview`](../backend/models.py) with `decision`, `review_target`, `reason_category`, `model_version`, `rag_version`, plus the original notes / edits / scores.
- Decisions are validated against [`REVIEW_DECISIONS`](../backend/services/review_constants.py): `approved` / `edited` / `rejected` / `unsafe` / `missing_evidence` / `wrong_escalation` / `needs_followup`.
- Reasons are validated against `REVIEW_REASON_CATEGORIES`.
- Aggregate counts feed the Safety & Eval Center's clinician feedback cards.
- The frontend ReviewPanel uses the same vocabulary via [`lib/constants.ts`](../frontend-react/src/lib/constants.ts).

## Single source of truth for safety strings

The backend [`review_constants.py`](../backend/services/review_constants.py) and the frontend [`lib/constants.ts`](../frontend-react/src/lib/constants.ts) define the same canonical strings for decisions, targets, reason categories, risk levels, confidence levels, and artifact statuses. The frontend file's header explicitly says "mirror of backend/services/review_constants.py — update both when changing values". Changing one without the other is a wire-incompatibility bug.

## Evaluation commands

```
python -m scripts.run_safety_eval_center               # safety + rag + drift
python -m scripts.run_safety_eval_center --only safety # one suite
python -m scripts.seed_demo_and_evals                  # full demo bootstrap
python -m pytest tests/test_safety_eval_center.py      # unit tests
python -m pytest tests/test_clinician_feedback.py      # review loop tests
```

## Honest limitations

- Training labels are synthetic; no real-world clinical validation has been performed.
- RAG metrics are heuristic proxies until labeled KB evaluation sets exist.
- ECE, calibration, drift, and subgroup numbers are engineering monitoring evidence — not clinical validity.
- The external real-data direction (BreastDCEDL/I-SPY1 MRI-derived baseline) currently yields AUROC ≈ 0.637, an honest indication that synthetic-trained models do not transfer to real MRI-derived signals without further work.

For more on what is and is not claimed, see [../MODEL_CARD.md](../MODEL_CARD.md), [../DATA_CARD.md](../DATA_CARD.md), and [../SAFETY_CARD.md](../SAFETY_CARD.md).
