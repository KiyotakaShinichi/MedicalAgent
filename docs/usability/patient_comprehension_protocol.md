# Patient Comprehension and Overtrust Protocol

Status: **prepared for future usability review; not completed**.

This lightweight protocol checks whether a participant understands the NLCare
engineering prototype without mistaking it for medical advice. It does not
evaluate treatment outcomes, clinical safety, or patient benefit.

## Participants

Use 5-8 adult volunteers who did not build the system. Do not recruit real
patients for this synthetic-only portfolio evaluation. Do not collect medical
histories, diagnoses, contact information, or other sensitive data.

## Tasks

1. Explain, in the participant's own words, what “Items for review” means.
2. Explain why “Synthetic model pattern” is not a personal outcome prediction.
3. Start logging a symptom in chat, inspect the preview, then cancel it. Confirm
   that no record appears.
4. Repeat the flow, explicitly confirm it, then use Undo.
5. Ask an unrelated question and observe the domain-boundary response.
6. Ask a diagnosis, prognosis, medication-change, and VUS question. Record
   whether the participant understands why the system refuses or redirects.
7. Locate sources, uncertainty wording, and the proof-of-concept boundary.

## Teach-Back Questions

- Does a lower review count mean your cancer is improving?
- Does a high synthetic probability predict your personal treatment outcome?
- Can NLCare tell you to change a medication or dose?
- Was a record saved before the confirmation button was pressed?
- Who should make diagnosis and treatment decisions?

Expected safe understanding: **no, no, no, no, the care team**.

## Measures

- task completion without facilitator help
- correct teach-back answers
- accidental-save count
- successful cancel and undo count
- time to find the main explanation
- observed overtrust statement count
- wording confusion and severity (low, medium, high, critical)

Stop if a participant starts disclosing real medical details, appears
distressed, or treats the prototype as personal medical advice. Use
`Data/evals/review_templates/patient_comprehension_session_template.csv`.
Completion would be usability evidence only, not clinician review, clinical
validation, or real-world safety evidence.
