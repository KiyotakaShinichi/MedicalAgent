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

## Privacy and security assumptions
- Designed with healthcare privacy principles in mind, but not certified or validated for clinical deployment.
- Demo role-based access only; production controls are not implemented.

## Known risks
- Synthetic data can hide real clinical complexity, bias, and missingness
- LLM adjudication can over- or under-block; deterministic guardrails remain primary

## Mitigations
- Non-diagnostic language throughout UI and docs
- Deterministic safety gates and refusal behavior
- Human-in-the-loop clinician review
- Audit logs and evaluation reports

## Clinical validation status
- Not clinically validated and not approved for clinical use.
