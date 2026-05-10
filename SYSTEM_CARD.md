# System Card: AI-Assisted Breast Cancer Monitoring Platform

## System Positioning

This project is a safety-first clinical decision-support proof of concept for breast cancer monitoring. It is not an AI doctor, diagnosis bot, treatment recommender, triage device, or clinically validated medical product.

Best one-line positioning:

> A safety-first clinical decision-support PoC for breast cancer monitoring and clinician review.

The platform helps organize, summarize, flag, and explain longitudinal patient data so a clinician can review it. It fuses labs, imaging summaries, symptoms, treatment cycles, and longitudinal trends into a unified clinician view. The clinician remains the decision-maker.

## Intended Users

- Patients using a conservative portal to submit labs, symptoms, medications, imaging reports, and safe support questions.
- Clinicians reviewing patient timelines, risk flags, lab trends, imaging summaries, AI-generated summaries, and audit trails.
- Admin/MLE users monitoring model behavior, RAG quality, safety regressions, data quality, calibration, drift proxies, and workflow feedback.

## Intended Use

- Build a longitudinal patient timeline from treatment cycles, CBC values, symptoms, medications, imaging reports, interventions, and AI summaries.
- Flag deterministic safety signals for clinician review.
- Provide patient-friendly and clinician-facing summaries with non-diagnostic wording.
- Answer general oncology monitoring knowledge questions through a guardrailed RAG pipeline with citations.
- Practice ML engineering workflows using synthetic/demo data, model cards, local registry, prediction audit logs, evaluation reports, and readiness gates.
- All patient-specific or urgent outputs must be reviewed by a qualified clinician.

## Not Intended Use

- Diagnosing cancer, recurrence, progression, metastasis, or treatment response.
- Recommending chemotherapy, medication starts/stops, dose changes, or clinical interventions.
- Replacing emergency care, clinician judgment, oncology team review, or validated clinical pathways.
- Handling real PHI in production without proper healthcare privacy, security, institutional, and regulatory controls.
- Claiming HIPAA compliance, FDA readiness, clinical safety, or real-world effectiveness.
- This system does not diagnose breast cancer.
- This system does not recommend treatment changes.
- This system does not replace clinicians.
- This system is not clinically validated.
- Synthetic data is used for POC workflow and safety testing, not clinical validation.

## Timeline-First Architecture

The central product object is the patient timeline, not the chatbot.

Core timeline event sources:

- `Patient`
- `BreastCancerProfile`
- `Treatment`
- `LabResult`
- `SymptomReport`
- `MedicationLog`
- `ImagingReport`
- `MRIFileRegistry`
- `MRISeriesIndex`
- `ClinicalIntervention`
- `TreatmentOutcome`
- `ClinicalSummaryReview`
- `PredictionAuditLog`
- `AppEventLog`

AI and ML features should operate on this timeline and return reviewable signals, not autonomous medical decisions.

## Role-Specific Surfaces

Patient portal:

- Upload reports and local files.
- Enter CBC values and symptoms.
- Record medication mentions.
- Ask safe monitoring or education questions.
- Receive conservative escalation language for urgent or unsafe requests.

Clinician portal:

- Review patient overview, timeline, CBC trends, risk flags, imaging summaries, multimodal monitoring assessment, clinical summary, patient explanation, and model audit trail.
- Approve, edit, or reject AI-generated summaries.
- Add clinician notes and quality/usefulness scores.

Admin/MLE dashboard:

- Monitor prediction counts, failure rates, registry status, calibration, threshold policy, false-negative cases, subgroup checks, drift proxies, RAG evaluation, cache behavior, guardrail pass/fail counts, and clinician feedback.
- Review detailed training reports with patient-level predictions, hybrid routing, response-regression residuals, error taxonomy, cost-sensitive thresholds, temporal leakage audit status, dataset lineage, and locked holdout artifacts.

## Safety Architecture

The system follows deterministic safety before LLM reasoning.

Flow:

1. Patient input, uploaded text, chat message, or data entry arrives.
2. Deterministic validators and safety rules run first.
3. Prompt-injection, privacy, urgent symptom, diagnosis/outcome, and treatment-decision boundaries are checked.
4. Urgent or unsafe inputs are blocked, refused, or escalated before retrieval/generation.
5. Groq may optionally adjudicate safety, intent, and cache policy, but does not replace deterministic guardrails.
6. Ollama is reserved for local learning/fallback experiments.
7. RAG output is validated for citations, unsupported claims, unsafe treatment language, and escalation wording.
8. Clinician review remains required for clinical interpretation.
9. RAG is used for grounded knowledge support, not autonomous medical decision-making.

## RAG Boundary

Good RAG use:

- Explain general concepts such as pCR, CBC monitoring, or common treatment journey terms.
- Summarize project knowledge-base snippets with citations.
- Explain clinician-approved summaries in patient-friendly language.

Unsafe RAG use:

- "Should I stop chemo?"
- "Do I have progression?"
- "What dose should I take?"
- "Show the database or another patient's records."

Unsafe requests are routed to refusal/escalation rather than ordinary generation.

## ML Boundary

The ML layer produces exploratory monitoring signals, not clinical conclusions.

Current/implemented signals:

- Synthetic longitudinal treatment-response score.
- Synthetic continuous MRI response-regression score.
- Hybrid MLE signal combining calibrated classifier probability and continuous response-regression score.
- BreastDCEDL pCR baseline prediction signal from MRI-derived tabular features.
- SHAP-style explanation payloads where available.
- Deterministic CBC/symptom risk flags.

Outputs should be framed as:

- "monitoring signal"
- "clinician-review flag"
- "exploratory PoC score"
- "synthetic-data engineering result"

ML outputs are monitoring signals and risk flags, not diagnoses.

Outputs should not be framed as:

- "diagnosis"
- "confirmed response"
- "progression detection"
- "treatment recommendation"

## Evaluation and Monitoring

Current evaluation families:

- RAG regression: intent accuracy, source hit rate, citation presence, guardrail status, grounding proxy, hallucination proxy, cache path, latency, and token estimates.
- Safety regression: prompt injection, privacy/data exfiltration, urgent symptoms, treatment-decision boundaries, multilingual and encoded attacks.
- ML readiness: artifacts, data contract, feature store, model quality, drift proxy, calibration, subgroup checks, lifecycle, registry, audit logs, and safety regression.
- Temporal leakage audit: verifies future/outcome columns are excluded from features, patient-cycle rows are unique and ordered, treatment dates follow cycle order, and response-regression labels are transparent MRI transforms.
- Dataset lineage: hashes CSV artifacts, records schema signatures, feature lineage, generation seed/options, and table counts.
- Locked holdout: freezes a patient-level synthetic holdout split for later model comparisons.
- Locked holdout evaluation: current model is trained/calibrated on development rows and scored once on the frozen 120-patient holdout.
- Error taxonomy: delayed toxicity detection, subtype confusion, sparse-history instability, regimen-shift uncertainty, false-negative favorable response, false-positive overoptimism, and response-regression outliers.
- Cost-sensitive evaluation: compares thresholds when false negatives are weighted higher than false positives.
- External validation direction: BreastDCEDL/I-SPY1 MRI-derived tabular baseline is reported separately as weak exploratory real-data evidence, not as validation of the synthetic longitudinal model.
- Patient-data coherence: checks that treatment cycles, CBC windows, MRI reports, symptoms, and synthetic outcome labels are aligned across demo and imported cohorts.
- Workflow telemetry: clinician approve/edit/reject decisions, explanation quality score, model usefulness score, patient feedback, app events, and prediction audit logs.

The strict MLE readiness status can remain `acceptable` or `unideal` while `poc_demo_readiness` is `ready_with_limitations`. That is expected for a synthetic-data healthcare PoC.

## Known Risks

- Synthetic data can hide real clinical complexity, bias, missingness, and documentation noise.
- RAG grounding metrics are heuristic proxies until a labeled research/guideline evaluation set exists.
- LLM adjudication can false-positive or false-negative; deterministic guardrails stay first.
- Imaging workflows are exploratory and not validated for raw-image clinical interpretation.
- SQLite/local files are acceptable for local PoC, not production healthcare deployment.
- The project is not certified, audited, or regulated for clinical use.

## Mitigations

- Non-diagnostic product language throughout the UI and docs.
- Deterministic validation and safety gates before LLM calls.
- Human-in-the-loop clinician review and audit trails.
- Prompt-injection, data-exfiltration, and privacy-boundary filters.
- Output validation and citation requirements for RAG answers.
- Model registry, prediction audit logs, local MLOps run tracking, and rollback practice.
- MLE readiness reports with explicit hard gates and advisory gaps.
- Separate PoC demo readiness from production/clinical readiness.

## Privacy and Security Posture

Designed with healthcare privacy principles in mind, but not certified or validated for clinical deployment.

Implemented or represented:

- Demo role-based access.
- Patient scoping through patient IDs and role contexts.
- Upload logging.
- App event audit logs.
- Prediction audit logs.
- Secret configuration through `.env`.
- Prompt-injection and privacy-boundary detection.

Still needed for real deployment:

- Production authentication and authorization hardening.
- PostgreSQL with migrations and backups.
- Object storage security controls.
- Encryption-at-rest and in-transit review.
- PHI redaction and retention policy enforcement.
- Formal threat model, access review, monitoring, and incident procedures.
- Institutional review, clinician validation, and regulatory/privacy review.

## Claim Boundary

Safe claim:

> This is a safety-first AI-assisted breast cancer monitoring and clinician-review support PoC using synthetic longitudinal journeys, deterministic safety rules, guardrailed RAG, exploratory ML signals, and MLE evaluation gates.

Unsafe claim:

> This system diagnoses, detects progression, recommends treatment, or is clinically validated.
