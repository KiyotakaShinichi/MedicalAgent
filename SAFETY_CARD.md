# Safety Card: MedicalAgent

MedicalAgent is a **safety-first** breast cancer **treatment monitoring and clinician-review support** platform. It is **not** an AI doctor, not a diagnosis bot, and not a treatment decision system. This card documents the deterministic safety boundaries, refusal rules, escalation logic, and clinician-in-the-loop design that the system commits to.

## Non-Diagnostic Boundaries

MedicalAgent **does not**:

- Diagnose cancer, recurrence, or metastasis.
- Confirm treatment response.
- Recommend, choose, change, or refuse medications, chemotherapy, dosing, schedules, surgery, radiation, or any other intervention.
- Interpret raw DICOM images for clinical decisions.
- Replace clinician judgment, hospital workflows, or regulated decision-support software.

The model patient-facing surface uses **monitoring and support language**, not diagnosis or directive language. Risk flags are framed as "for clinician review", never as conclusions.

## Refusal Rules

The pipeline routes queries into one of several deterministic intents before LLM adjudication runs. When a request lands in any of these intents, the pipeline returns a structured refusal/escalation message rather than free-form medical advice:

| Intent                           | Behavior                                                                                |
|----------------------------------|------------------------------------------------------------------------------------------|
| `safety_boundary`                | Refuses diagnosis, outcome prediction, prognosis, or "do I have cancer" style questions. |
| `treatment_decision_boundary`    | Refuses medication, dose, regimen, schedule, or treatment-change requests.               |
| `security_boundary`              | Blocks prompt injection, cross-patient access attempts, system-prompt extraction.        |
| `urgent_symptom_escalation`      | Escalates to "contact your care team / emergency services" for crisis-style symptoms.    |
| `insufficient_evidence_refusal`  | Declines to answer when KB and patient record provide no supporting evidence.            |

Refusals are paired with **escalation suggestions**: "contact your clinician", "log the symptom for review", "call emergency services", or "use the upload feature to share the report".

## Medication / Treatment Decision Boundaries

- Medication terms (`paclitaxel`, `tamoxifen`, `trastuzumab`, `capecitabine`, `dose`, `medication`, etc.) trigger the `medication_refusal` refusal type.
- Treatment-change requests trigger the `treatment_refusal` refusal type and route to clinician review.
- The agent never produces dose numbers, schedule changes, or substitution suggestions, even when asked indirectly.

## Privacy Boundaries

- Patient context is always scoped by the authenticated patient ID.
- The agent rejects cross-patient queries (e.g. "show me records for patient X") at the input guardrail.
- Audit logs record every clinician access to a non-assigned patient.
- Uploaded files are stored under the requesting patient's namespace; no automatic cross-patient sharing.

## Urgent Symptom Escalation

Deterministic rules in [`backend/processing/risk_engine.py`](backend/processing/risk_engine.py) catch:

- WBC, ANC, hemoglobin, and platelets below configurable urgent thresholds (`config/safety_thresholds.yaml`).
- Large drops from baseline (e.g. WBC drop ≥ 50% from baseline).
- High symptom severity, fever language, breathlessness, chest pain, suicidal ideation, etc.

When triggered, the agent issues an escalation message instructing the patient to contact their care team and, for crisis-style language, includes regional emergency-services guidance. Escalations are logged with full evidence (which threshold was crossed, which value triggered the rule, threshold-config version).

## Crisis Handling

For messages matching crisis patterns (self-harm language, severe distress, suicidal ideation), the agent:

1. Does **not** attempt counseling or therapeutic intervention.
2. Returns a safety message pointing to emergency services and crisis hotlines.
3. Flags the conversation for clinician review.
4. Logs the event with `safety_level=urgent` for the audit trail.

## Prompt-Injection Defenses

The safety red-team suite (`backend/services/safety_red_team.py`) regularly exercises:

- Direct instruction override ("ignore previous instructions…").
- Indirect injection via uploaded notes / report text.
- Encoded / base64 / unicode-confusable injection.
- Multilingual injection (English + Tagalog mixing).
- Privacy-exfiltration prompts ("dump all patients", "show the system prompt").
- Fake-report prompts asking for diagnosis from synthetic MRI/CT/ultrasound text.

Defenses are layered:

1. **Input guardrail** — pattern-based detection of known injection shapes; blocked messages never reach retrieval or generation.
2. **Intent router** — deterministic routing to `security_boundary` / `safety_boundary` based on detected patterns.
3. **Output guardrail** — last-mile check that no PII from non-assigned patients and no medical-directive phrasing leaks into the reply.
4. **Audit log** — every blocked request is recorded with the matched pattern and the actor role.

## Deterministic vs. LLM Safety

Safety-critical behavior is **never** delegated to LLM judgment alone. Specifically:

- Medication, treatment-decision, and diagnosis refusals come from deterministic routing, not from an LLM "deciding to refuse".
- CBC and symptom escalation use rule thresholds, not LLM inference.
- Prompt-injection blocks are pattern-driven; the LLM is not asked to "spot the injection".
- The LLM is used for *grounded explanation* and *patient-friendly phrasing*, after deterministic guardrails decide the route.

## Clinician-in-the-Loop Design

- Every AI-generated summary or risk flag is routed into the clinician review queue.
- Clinicians can mark each item as `approved`, `edited`, `rejected`, `unsafe`, `missing_evidence`, `wrong_escalation`, or `needs_followup`.
- Edits, rejections, and unsafe flags are logged with reason categories and feed the Safety & Evaluation Center.
- The dashboard's *Clinician feedback loop* surfaces these counts so governance can see, e.g., a rising `wrong_escalation` rate before it becomes a pattern.

## Uncertainty Communication

Every risk flag includes an `uncertainty` block with `confidence_level` (low / moderate / high), `uncertainty_reason`, `missing_data_indicators`, and `clinician_review_required`. Patient-facing wording surfaces this as text such as:

> Estimated response signal: 0.68. **Confidence: moderate.** Uncertainty is elevated because the most recent MRI follow-up is missing and the recent CBC trend is incomplete. **Clinician review required.**

The agent never reports a risk flag without uncertainty wording.

## What This Card Is Not

- This card is **not** a regulatory submission.
- It is **not** a clinical validation document.
- It documents **engineering safety design**, the test cases that exercise it, and the failure modes that remain unresolved.

For the full list of model and data assumptions and limitations, see [MODEL_CARD.md](MODEL_CARD.md) and [DATA_CARD.md](DATA_CARD.md).
