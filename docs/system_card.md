# System Card: MedicalAgent

## Intended users
- Patients using a conservative portal to submit labs, symptoms, medications, imaging reports, and safe support questions.
- Clinicians reviewing patient timelines, risk flags, lab trends, imaging summaries, and AI-generated summaries.
- Admin and MLE users monitoring model behavior, RAG quality, safety regressions, and workflow feedback.

## Intended use
- Build a longitudinal patient timeline from treatment cycles, CBC values, symptoms, medications, imaging reports, interventions, and AI summaries.
- Surface monitoring signals and clinician-review flags with non-diagnostic wording.
- Answer general oncology monitoring questions through a guardrailed RAG pipeline with citations.
- Practice ML engineering workflows using synthetic data, model cards, local registry, prediction audit logs, evaluation reports, and readiness gates.
- All patient-specific or urgent outputs must be reviewed by a qualified clinician.

## Not intended use
- Diagnosing cancer, recurrence, progression, metastasis, or treatment response.
- Recommending chemotherapy, medication starts or stops, dose changes, or clinical interventions.
- Replacing emergency care, clinician judgment, oncology team review, or validated clinical pathways.
- Claiming HIPAA compliance, FDA readiness, clinical safety, or real-world effectiveness.
- This system does not diagnose breast cancer.
- This system does not recommend treatment changes.
- This system does not replace clinicians.
- This system is not clinically validated.
- Synthetic data is used for POC workflow and safety testing, not clinical validation.

## Non-diagnostic boundary
- RAG is used for grounded knowledge support, not autonomous medical decision-making.
- ML outputs are monitoring signals and risk flags, not diagnoses.

## Major components
- Patient, clinician, and admin portals
- Timeline and risk processing services
- Deterministic safety and guardrails
- Guardrailed RAG pipeline with hybrid retrieval
- ML training, evaluation, registry, and readiness gates
- Audit logs and evaluation reports

## Safety mechanisms
- Deterministic safety and privacy guardrails before retrieval or generation
- Prompt-injection detection and refusal
- Output validation for treatment directives and diagnosis claims
- Cache policy that blocks patient-specific and urgent content

## Human oversight
- Clinician review workflow with approve, edit, reject, and follow-up logging
- Audit logs for summary reviews and feedback

## RAG limitations
- Grounding and hallucination metrics are heuristic proxies until labeled KB data exists
- Citations support educational content but do not replace clinician judgment

## ML limitations
- Synthetic data does not prove real clinical performance
- Imaging workflows use report text or tabular features, not validated raw-image models
- Temporal, noise, and calibration reports are simulator stress tests only. They measure engineering robustness, not clinical deployment robustness.

## Privacy and security assumptions
- Designed with healthcare privacy principles in mind, but not certified or validated for clinical deployment.
- Demo role-based access only; production controls are not implemented.

## Known risks
- Synthetic data can hide real clinical complexity, bias, and missingness
- LLM adjudication can over- or under-block; deterministic guardrails remain primary

## Harm modes and mitigations

| Failure mode | Possible harm | Mitigation in this PoC | Residual risk |
|--------------|---------------|------------------------|---------------|
| Missed urgent symptom escalation | Patient delays contacting care team | Deterministic urgent-word and CBC safety rules before LLM/RAG | Keyword and multilingual paraphrase gaps remain |
| Incorrect treatment-response framing | Patient interprets a monitoring signal as diagnosis | Non-diagnostic templates, refusal boundaries, clinician review | UI scores can still feel overly clinical |
| Supplement interaction answer is too permissive | Patient starts supplement that interacts with therapy | Supplement-safety KB category, treatment-change refusal, oncology-team prompt | Source coverage must stay curated and current |
| RAG retrieves outdated or low-authority content | Educational answer may be misleading | Source trust levels, citation validation, source-hit regression tests | KB freshness requires ongoing review |
| Subgroup calibration gap hidden by global metric | Model may be less reliable for a subgroup | Per-subgroup ECE plus performance metrics in benchmark artifacts | Synthetic subgroup distributions do not prove real fairness |
| Partial chat entry saved incorrectly | Patient record contains wrong symptom/lab | Honest-save rules, required severity for symptom saves, DB flush before confirmation | Multi-turn ambiguity still requires review |

## Mitigations
- Non-diagnostic language throughout UI and docs
- Deterministic safety gates and refusal behavior
- Human-in-the-loop clinician review
- Audit logs and evaluation reports
- Admin/MLE dashboard surfaces temporal generalization, high-noise robustness, and calibration comparison reports with synthetic-data caveats.

## Clinical validation status
- Not clinically validated and not approved for clinical use.

---

## Engineering evaluation summary (as of 2026-05)

All metrics below are on **synthetic data** unless noted.

| Evaluation | Result | Data type |
|------------|--------|-----------|
| Agent regression suite | 45/45 pass (100%) | Synthetic eval cases |
| Attack block rate | 1.0 (100%) | Security eval cases |
| Expected source hit rate | 1.0 (100%) | Education eval cases |
| RAG grounding score | ~0.91 avg | Synthetic eval cases |
| Hallucination risk | ~0.26 avg | Heuristic proxy |
| MLE readiness | Acceptable | Synthetic pipeline |
| Safety regression gate | Strong | Synthetic regression |
| Calibration (ECE) | Reported | Synthetic holdout |
| RAG ablation (BM25 vs sparse vs dense hybrid vs reranked) | Reported - full reranked pipeline tracked | Synthetic education cases |
| Per-prediction error table (TP/FP/TN/FN) | Reported - sensitivity, specificity, MAE, SHAP | Synthetic holdout |
| Noise robustness (5 perturbation modes) | Reported - per-mode AUROC/sensitivity degradation | Synthetic perturbations |
| Temporal generalization (timeline + cycle splits) | Reported - generalization gap vs random baseline | Synthetic timeline |
| External validation | Direction only - BreastDCEDL / I-SPY1 tabular features | Real MRI datasets |

**Important:** synthetic AUROC, Brier score, and ECE are expected to be favourable because the training and test distributions are generated by the same procedure. External validation on BreastDCEDL/I-SPY1 provides a directional signal only and does not constitute clinical validation.

## RAG pipeline technical summary

| Stage | Implementation |
|-------|---------------|
| Retrieval | Dense sentence-transformer + FAISS with BM25/RRF when available; sparse BM25 + TF-IDF fallback |
| Source boost | Curated-source boost (NCI, CDC, ACS, Project KB) in hybrid_retrieval scoring |
| Window expansion | Parent-child chunk window expansion |
| Reranking | Coverage + safety + source boosts in rerank_context |
| Compression | Contextual compression on top-K |
| Cache | Semantic similarity cache (threshold 0.86), blocked for patient-specific/urgent content |
| LLM adjudication | Optional local LLM for ambiguous routing; deterministic gate takes precedence |

## Cost-sensitive evaluation design

This system assumes **FN cost > FP cost** in the cancer monitoring context:
- A missed non-response (false negative) may delay clinician intervention.
- An unnecessary flag (false positive) increases review burden but does not harm patients directly.
- The classification threshold is set below 0.50 (engineering heuristic) to favour sensitivity.
- Weighted cost metric: FN_weight = 3, FP_weight = 1 (PoC heuristic, not a clinical guideline).
