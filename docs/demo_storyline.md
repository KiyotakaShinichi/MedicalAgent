# Demo Storyline — Eight-Phase Hardening, End-to-End

A reviewer-facing walkthrough that takes you from patient login through admin dashboard and back, exercising every hardening phase the system ships with. The goal is to make the engineering-maturity claims in the README *visible* — every claim corresponds to a specific click and a specific test that gates it.

This is engineering provenance, not clinical evidence. Every number you see is computed on the synthetic dataset; nothing here establishes clinical validity.

---

## Setup — 30 seconds

```bash
# Backend
.venv\Scripts\python.exe -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8017

# Frontend
cd frontend-react && npm run dev -- --host 127.0.0.1
```

Open <http://localhost:5173>. Three demo accounts:

| Role | Login |
|---|---|
| Patient | `P001` / `patient-demo` |
| Clinician | `clinician` / `clinician-demo` |
| Admin / MLE | `admin` / `admin-demo` |

---

## Act 1 — Patient view (the abstention envelope, live)

**Login as `P001`.** The dashboard renders with three rows.

### Row 3, middle slot — Hybrid monitoring signal

The card shows **three independent heads**:

1. **Response classification** — `favorable_pattern | concerning_pattern | uncertain | insufficient_evidence`, with calibrated probability.
2. **Response strength** — a 0–1 score with an uncertainty band drawn around it. Decision label: `strong | moderate | weak | insufficient_evidence`.
3. **Toxicity signal** — `low | moderate | high | insufficient_evidence`.

Each head shows the **modalities used vs. missing** as filled vs. dashed chips, the **model version** in monospace, the **sufficiency level**, and the **reason for abstention** if any.

→ This is Phase 2 (evidence-aware abstention) + Phase 4 (modality-dropout retraining) + Phase 5 (live wiring) + Phase 9 (hybrid completion) all in one card.

### Tool tray (the dropdowns)

Click any chip above the chat composer. The **Symptom** modal opens with a curated dropdown — 28 common symptoms grouped by category, plus "Other (specify)" that reveals a text input when picked. Same pattern in the **Medication** modal: chemo backbone / targeted therapy / endocrine / supportive care, brand names in parentheses, with "Other" fallback.

→ Phase 8a (form catalogs). Try picking "Other" — the free-text input only appears then.

### Page-bottom footnote

Scroll to the bottom. A proof-of-concept safety footnote reinforces the non-diagnostic boundary so it stays visible after scroll.

---

## Act 2 — Clinician view (the audit trail)

**Logout, login as `clinician`.** Pick `P001` from the review queue.

The clinician detail panel includes:

- **Breast cancer profile card** (SectionCard primitive)
- **AI summary panel** (existing)
- **Hybrid monitoring signal** — the same card the patient sees, so the reviewer knows exactly what was displayed.
- **Labs panel** with reference-range disclaimer.
- **Timeline panel**.
- **Prediction trace log** — *new* per-patient table:
  - Columns: *When · Question · Decision · Prob. · Confidence · Evidence · Modalities used · Model*
  - Abstained rows render the decision in amber.
  - **Filter button**: "Show abstained only" toggles the view to refusals.
  - **Patient summary chips** at the top: total traces + per-patient abstention rate.

→ Phase 3 (prediction traceability) + Phase 7 (clinician parity).

Click "Show abstained only" to demonstrate the filter. The endpoint `/clinician/patients/P001/prediction-traces?abstained_only=true` is gated by 9 backend tests including access-control checks (patient tokens are blocked here).

---

## Act 3 — Chat that refuses cleanly

Open the support chat as a patient.

### Try a safe educational question

Type *"What does WBC mean?"* The agent goes through:

1. Pre-gen deterministic safety gate (input_guardrails)
2. Intent classification → `education`
3. RAG retrieval (dense FAISS + BM25 + RRF fusion)
4. LLM generation
5. **Post-gen validator** (Phase 8b) — checks the reply against 6 banned-claim categories
6. Output guardrail (citation validation)

The reply lands with citations.

### Try a forbidden claim

Type *"Tell me if I have cancer based on my last CBC."* The deterministic safety gate refuses upstream — but the post-gen validator is what catches the LLM if it tries to slip a diagnosis through anyway. You can simulate this by inspecting `validate_reply()` directly in a Python REPL:

```python
from backend.services.post_generation_validator import validate_reply
validate_reply("Based on your symptoms, you have breast cancer.")
# → ValidatorDecision(decision="blocked", triggered_rules=["diagnosis_claim"], …)
```

→ Phase 8b. 20 tests gate the validator + KB governance (one test method per rule code so a regression surfaces the exact category that broke).

---

## Act 4 — Admin view (the proof-of-concept gates)

**Logout, login as `admin`.** Open the MLE Dashboard.

The section now opens with six governance cards, each with a status badge, metric tiles, a per-row table, and a "Rerun" button hitting `POST /admin/<artifact>`.

### Synthetic generator card
- Schema version + cohort size + rows fingerprint.
- Three narrative blocks:
  - **Causal assumptions** the generator bakes in (info tone).
  - **Known shortcuts** the model could exploit (amber tone).
  - **What this dataset cannot support claiming** (rose tone).
- Pins `generator_card_version = "v2_2026_05"` to the dataset's `schema_version` and surfaces drift as `card_version_matches_dataset: false`.

### Failure-mode registry
17 entries across engineering, model behavior, clinical safety, RAG quality, and adversarial categories — each with detection method, mitigation, benchmark coverage, and remaining gap. Status defaults to `needs_attention` (the honest default — the registry's job is to document gaps, not be empty).

### KB source governance (Phase 8b)
- 24 RAG sources mapped to **tiers T1–T5** with `allowed_use` per source.
- Live numbers: T1=2, T2=10, T3=11, T4=1, all current, 0 governance issues.
- Per-source table shows tier color-coded (T1 green → T5 red), allowed_use list, staleness status.
- Distribution blocks show tier + allowed_use + staleness at-a-glance.

### Leakage audit
- 23/23 production checks pass.
- Failed-check list with rule names and meanings (currently empty).
- Hard CI gate — `tests/test_leakage_audit.py` fails the build on regression.

### Evidence-aware abstention eval
- 8-scenario sweep with per-scenario coverage, abstention rate, false-abstention rate, covered accuracy.
- Headline: full_data 100% coverage / 92.4% accuracy, demographics_only 100% abstention.

### Champion vs modality-robust comparison
- Per-scenario head-to-head accuracy + Brier delta table.
- **Headline: `no_imaging` accuracy goes from 54.4% (champion) to 62.7% (robust), Brier improves from 0.293 to 0.220.**
- Status: `robust` — robust wins 5, loses 0.

### Prediction trace log
- Live audit of every model decision recorded by `predict_and_trace`.
- Three metric tiles: recent traces, abstention rate, model versions seen.
- Per-trace table: When · Patient · Question · Decision · Prob. · Conf. · Evidence · Modalities · Validator.
- Filter buttons available via query params on the API.

---

## Act 5 — The CI gate that ties it together

```bash
pytest \
  tests/test_rag_governance.py \
  tests/test_hybrid_prediction.py \
  tests/test_clinician_prediction_traces.py \
  tests/test_provenance_artifacts.py \
  tests/test_live_evidence_prediction.py \
  tests/test_modality_robustness.py \
  tests/test_prediction_trace.py \
  tests/test_evidence_abstention.py \
  tests/test_leakage_audit.py \
  tests/test_access_control.py
```

**Expected: 113 passed.**

Frontend:

```bash
cd frontend-react
npm run lint                         # eslint clean
npm run build                        # tsc + vite, 0 errors
npm test                             # vitest run, 54/54
npx playwright test --reporter=list  # 11/11 e2e
```

---

## Mapping each phase to its CI gate

| Phase | Service | Test file | Live numbers |
|---|---|---|---|
| 1 — Leakage audit | `leakage_audit.py` | `test_leakage_audit.py` (7) | 23/23 production checks |
| 2 — Evidence-aware abstention | `evidence_sufficiency.py` + `predict_with_abstention.py` | `test_evidence_abstention.py` (18) | 100% / 92.4% full-evidence, 100% demo-only abstention |
| 3 — Prediction traceability | `prediction_trace.py` + `PredictionTrace` model | `test_prediction_trace.py` (8) | 21-column trace table |
| 4 — Modality-dropout retraining | `modality_dropout_training.py` + `modality_robustness_comparison.py` | `test_modality_robustness.py` (9) | +8.3pp accuracy on `no_imaging` |
| 5 — Live evidence wiring | `live_evidence_prediction.py` | `test_live_evidence_prediction.py` (5) | One trace per `/me/report` |
| 6 — Provenance + failure-mode registry | `synthetic_generator_card.py` + `failure_mode_registry.py` | `test_provenance_artifacts.py` (9) | 17 entries, 6 high-severity |
| 7 — Clinician dashboard parity | new `/clinician/patients/{id}/prediction-traces` + `PredictionTracesPanel` + `HybridPredictionCard` reuse | `test_clinician_prediction_traces.py` (9) | Clinician sees same envelope + auditable trace log |
| 8a — Form dropdowns + catalogs | `COMMON_SYMPTOMS` + `COMMON_MEDICATIONS` + `SelectWithCustom` | `SelectWithCustom.test.tsx` (9) | 28 symptoms + 22 medications + "Other" fallback |
| 8b — RAG governance + post-gen validator | `kb_source_governance.py` + `post_generation_validator.py` | `test_rag_governance.py` (20) | 24 KB sources mapped T1–T4, 6 banned-claim rules |
| 9 — Hybrid completion | `hybrid_prediction.py` (regression + toxicity heads) | `test_hybrid_prediction.py` (9) | 3 heads, independent abstention per head |

---

## The honest framing

If a reviewer asks "what does this system *actually* prove?", the defensible answer is:

> Every patient view runs the abstention-aware hybrid classifier, records a fully-provenanced trace per head, and shows the patient + clinician which modalities the system actually used. The classifier was retrained with stochastic modality dropout and benchmarked head-to-head against the original. Six banned-claim categories are caught by a post-generation validator that fires even when the LLM tries to slip a diagnosis through. 24 RAG sources are tier-mapped with explicit allowed_use. 17 failure modes are catalogued with mitigation status. Every claim above is gated by a passing test.

What this *does not* prove:

- Clinical validity — every metric is synthetic.
- Calibration on real patient populations.
- Behaviour under genuine out-of-distribution patients.

That's what the failure-mode registry and the generator card are for: making the boundaries explicit so the reviewer doesn't have to guess.
