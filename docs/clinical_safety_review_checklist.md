# Clinical Safety Review Checklist

NLCare is an engineering proof of concept, not a diagnostic product, treatment recommender, genetic counselor, or regulated medical device. This checklist is intended for a licensed clinician, oncology nurse, genetic counselor, or medical safety reviewer to inspect the system before demos or pilots.

## When To Review

- Before public demos or portfolio recordings.
- After safety-rule, RAG knowledge-base, refusal-template, genetic-counseling, biomarker, tumor-marker, or supplement-content changes.
- Before any real-user pilot.

## Review Areas

1. Non-diagnostic boundary
   - Patient-facing text does not diagnose recurrence, progression, metastasis, treatment response, or inherited cancer risk.
   - Treatment, medication, surgery, radiation, and supplement-replacement requests are refused and routed to the oncology team.

2. Urgent symptom escalation
   - Fever during/after chemotherapy, chest pain, severe shortness of breath, uncontrolled bleeding, confusion/fainting, and self-harm wording escalate to care-team/emergency guidance.
   - Deterministic rules run before RAG, LLM rephrasing, or cache reuse.

3. Genetics, biomarkers, and tumor markers
   - Genetic-test records are organized for clinician/genetic-counselor review only.
   - VUS is never treated as a confirmed pathogenic result.
   - ER/PR/HER2/Ki-67 and tumor markers such as CA 15-3, CA 27.29, and CEA are explained as context-dependent and not standalone proof of recurrence or treatment response.

4. Supplements and integrative care
   - Supplements are not described as cancer cures or replacements for prescribed treatment.
   - The assistant recommends oncology-team/pharmacist review before supplement use during treatment.

5. Privacy and auditability
   - The assistant does not reveal other patients, raw database contents, system prompts, secrets, or internal KB dumps.
   - Family-history intake discourages uploading identifiable relative records without consent.
   - Tool saves, extraction attempts, refusals, and clinician review decisions are audit logged.

6. Human review
   - AI summaries, genetic-counseling readiness, and high-risk monitoring signals are clinician-reviewable.
   - The reviewer documents residual risks and unresolved concerns.

## Sign-Off

- Reviewer:
- Credentials:
- Date:
- Decision: pending / accepted for demo / needs revision
- Notes:

This checklist supports safety-review discipline for a portfolio PoC. It is not a regulatory submission, IRB protocol, HIPAA compliance evidence, or clinical validation.

