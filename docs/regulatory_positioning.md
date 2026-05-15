# Regulatory Positioning (PoC)

This project is a safety-first clinical decision-support PoC. It is not a medical device and is not clinically validated.

## Intended use
- Summarize longitudinal monitoring signals for clinician review
- Provide patient-friendly explanations with non-diagnostic language
- Support evaluation and audit workflows for ML engineering practice

## Not intended use
- Diagnosis, prognosis, or treatment selection
- Clinical triage or emergency decision-making
- Replacement for clinician judgment

## Evidence packages (engineering only)
- System card, model cards, data card
- RAG and safety regression suites
- MLE readiness gates and evaluation reports
- Audit logs and feedback metrics

## Gaps for regulatory readiness
- Clinical validation study
- Formal risk management documentation
- Security and privacy compliance audits
- Quality management system (QMS) processes

## SaMD classification rationale if deployed

If this PoC were converted into a real product, it would likely need formal
Software as a Medical Device (SaMD) analysis because it processes patient-
specific oncology monitoring data and surfaces review recommendations. A
conservative deployment assumption is:

- United States: likely clinical decision support software requiring legal and
  regulatory review under FDA CDS/SaMD guidance, especially if outputs are not
  independently reviewable by clinicians.
- European Union: likely MDR Rule 11 software because it provides information
  used to support clinical monitoring decisions; classification would depend on
  intended use and risk of missed deterioration.

This repository does not claim FDA clearance, CE marking, HIPAA compliance, or
clinical validation. The current engineering boundary is: synthetic/demo data,
non-diagnostic monitoring support, clinician-in-the-loop review, and explicit
not-for-clinical-use labeling.

## Clinical review plan

Before any real clinical pilot, the following assets should be reviewed by a
licensed oncology clinician or oncology nurse practitioner:

- Refusal and escalation templates.
- Urgent symptom and CBC safety thresholds.
- Symptom keyword lists, including multilingual/code-switched examples.
- Patient-facing summaries and reading level.
- Supplement/interactions answers.
- Imaging-summary language for MRI, ultrasound, and CT reports.
- Known failure modes and mitigation table in the system card.

No clinical advisor has signed off on this PoC yet. That is a known limitation,
not a hidden assumption.
