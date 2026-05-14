# MedicalAgent — Demo Script

A 5-minute click-by-click walk-through that surfaces the safety, evaluation, and clinician-review story without making clinical claims.

**Important:** this is a synthetic-data portfolio system. Numbers shown in the dashboard are engineering monitoring evidence, not clinical validation. See [SAFETY_CARD.md](SAFETY_CARD.md) and [MODEL_CARD.md](MODEL_CARD.md).

## 0. One-time setup (≈ 2 min)

```bash
# Backend
python -m venv .venv
.venv\Scripts\activate           # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt

# Frontend
cd frontend-react
npm install
cd ..

# Seed demo patients + regenerate Safety & Eval Center artifacts
python -m scripts.seed_demo_and_evals
```

Start the stack in two terminals:

```bash
# Terminal 1 — backend
python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8017 --reload

# Terminal 2 — frontend
cd frontend-react
npm run dev
```

Open <http://localhost:5173>.

## 1. Patient view — "How does the patient see AI output?" (≈ 60s)

1. On the login screen, click the **Patient** demo pill and log in as `P001` / `patient-demo`.
2. The patient banner shows monitoring status — **not** a diagnosis.
3. Scroll to **Treatment Timeline**.
   - Lab and symptom events are plain records.
   - Any AI risk flag is rendered with the canonical `AIGeneratedLabel` row: a purple "AI-generated" chip, a confidence chip (`Confidence: moderate`), an amber "Clinician review required" badge, the source/model/timestamp metadata line, and an italic uncertainty reason sentence.
4. Open **Support Chat** and ask: *"Do I have cancer?"*
   - The agent refuses, routes to the diagnosis-refusal safety boundary, and includes the standard "Not a diagnosis. For clinician review only." disclaimer.
   - The reply is logged in the audit trail; the chat panel never echoes diagnostic vocabulary.
5. Ask: *"Should I stop my chemotherapy dose?"*
   - The agent refuses with the medication-refusal route and routes the conversation to clinician review.

> Recruiter takeaway: the patient surface is non-diagnostic; every AI output ships with confidence, uncertainty wording, and a "Clinician review required" badge.

## 2. Clinician view — "How does clinician review work?" (≈ 90s)

1. Log out, return to the login screen, choose **Clinician** and sign in as `clinician` / `clinician-demo`.
2. The left rail shows the review queue, priority-sorted by urgent flag count + missing-data warnings + lack of recent review.
3. Click a queued patient.
4. The right pane shows the patient's evidence: AI summary, lab trends, timeline (including the same `AIGeneratedLabel` row), and `ReviewPanel`.
5. In the review panel:
   - Optionally pick a **Reason category** (e.g. "Missing imaging follow-up").
   - Click any of the seven canonical decisions: **Approve**, **Save edit**, **Needs follow-up**, **Request more evidence**, **Wrong escalation level**, **Reject**, **Mark unsafe**.
   - The decision is posted to `POST /patients/{id}/summary-review` with `decision`, `reason_category`, `model_version`, and `rag_version`. The audit trail captures everything.

> Recruiter takeaway: clinician-in-the-loop is real, the decision vocabulary is enumerated and logged, and reviews are pure audit data — they never silently change the patient record.

## 3. Admin / MLE view — "How is the system measured?" (≈ 2 min)

1. Log out, sign in as `admin` / `admin-demo`.
2. The admin dashboard opens on **Overview**. Skim it for orientation.
3. Click the **Safety & Eval Center** tab. Each card has graceful empty / `not_generated` states; after `seed_demo_and_evals.py` the cards are populated.
   - **Safety red-team suite** — pass rate, failed-case list, category breakdown for prompt injection, urgent symptom escalation, medication refusal, cross-patient privacy.
   - **RAG evaluation** — citation coverage, expected-source hit, refusal correctness, average grounding score, average hallucination score, retrieval P@3.
   - **Calibration** — best model, Brier, pre-/post-temperature ECE.
   - **Drift & data quality** — `data_source: synthetic_demo` label, missing CBC rate, completeness score, lab distribution shift, imaging keyword shift, subgroup positive-rate shift.
   - **Clinician feedback loop** — counts of approved / edited / rejected / unsafe plus reason-category counts.
   - **Failure case gallery** — 8 representative failure modes with `what happened`, `why it's risky`, `system response`, `mitigation`, `unresolved`.
4. Click **Re-run** on any suite to regenerate the artifact live.
5. Open the **MLE Gates** tab to see the locked-holdout numbers; remember these are on synthetic data.
6. Open the **Imaging MLE** tab for the BreastDCEDL / ultrasound / CT-lesion direction.

> Recruiter takeaway: the project ships honest engineering monitoring evidence (red-team, RAG eval, drift, calibration, clinician feedback, failure modes) and labels its synthetic basis at every step.

## 4. Honest limitations to mention in conversation

- Training data is **synthetic**. The champion AUROC on synthetic data is excellent (≈ 0.96 on the locked holdout); the **real-data direction** (BreastDCEDL/I-SPY1) baseline yields AUROC ≈ 0.637.
- RAG grounding / hallucination metrics are heuristic proxies, not labeled-gold scores.
- No PHI / regulatory controls. This is a portfolio PoC, not a clinical-grade system.
- One pre-existing failing test (`test_breast_monitoring.py::test_agent_regression_suite_tracks_guardrails_and_sources`) sits in HEAD; the safety red-team suite covers the same behavior under a different metric.

## 5. Optional deeper dives

| Topic | Where to point |
|---|---|
| Safety design | [SAFETY_CARD.md](SAFETY_CARD.md) |
| Model & metrics | [MODEL_CARD.md](MODEL_CARD.md) |
| Data assumptions | [DATA_CARD.md](DATA_CARD.md) |
| Pipeline architecture | [docs/AI_ML_ARCHITECTURE.md](docs/AI_ML_ARCHITECTURE.md) |
| Role-scoped flows | [docs/ROLE_BASED_FLOWS.md](docs/ROLE_BASED_FLOWS.md) |
| Evaluation & safety story | [docs/EVALUATION_AND_SAFETY.md](docs/EVALUATION_AND_SAFETY.md) |
| Frontend architecture | [docs/frontend_architecture.md](docs/frontend_architecture.md) |

## 6. Screenshot placeholders

Drop screenshots into `docs/screenshots/` and uncomment the references below. (The repo intentionally ships without bundled screenshots so the demo always shows live data, not stale captures.)

- `docs/screenshots/patient_timeline_with_ai_label.png` — patient TimelinePanel with `AIGeneratedLabel`.
- `docs/screenshots/clinician_review_panel.png` — clinician ReviewPanel with the 7-decision row + reason category.
- `docs/screenshots/safety_eval_center.png` — admin Safety & Eval Center with all cards populated.
- `docs/screenshots/drift_subgroup_panel.png` — subgroup performance drift block.

When recording a demo video, run through sections 1 → 2 → 3 in that order — that produces the cleanest "safe AI for clinician review" narrative.
