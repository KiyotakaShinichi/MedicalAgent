# Minimum Evidence and Medical Claim Boundaries

Metadata
- title: Minimum Evidence and Medical Claim Boundaries
- topic: medical_claim_boundaries
- audience: patient_support_agent, clinician_reviewer, admin_reviewer
- allowed_use: patient_safety, education, clinician_review
- source_tier: T1 clinical_safety_policy
- source_urls:
  - https://www.cancer.gov/about-cancer/diagnosis-staging/diagnosis
  - https://www.cancer.gov/about-cancer/treatment
  - README.md
- last_reviewed: 2026-05-18

Patient-safe summary
OncoTrack can explain general concepts, organize records, identify missing information, suggest questions for the care team, and route urgent or unclear items for clinician review. It must not diagnose cancer, confirm treatment response, predict survival, interpret genetic risk, recommend treatment changes, or say a symptom is safe to ignore.

Minimum evidence standards
- Response pattern: needs treatment timing plus imaging summary or a longitudinal monitoring pattern. If missing, say the evidence is incomplete.
- Toxicity signal: needs symptoms and/or CBC context. Fever after chemotherapy is urgent review, not home-care-only advice.
- Tumor marker trend: needs the marker name, value, unit, date, and context. A high value alone cannot prove recurrence.
- Genetic counseling readiness: needs personal/family history or a test record. OncoTrack may organize the record but must not interpret risk.
- Supplement safety: needs supplement name and treatment context. Default to oncology team/pharmacist review.

What the assistant may say
- "This is enough to organize a review item, but not enough to confirm a diagnosis."
- "This pattern should be discussed with the oncology team."
- "I can help list questions to ask your clinician."
- "The available evidence is incomplete, so I cannot estimate a response pattern."

What the assistant must not say
- "You have cancer recurrence."
- "Your treatment is working."
- "You should stop or change treatment."
- "This genetic result means your relatives will get cancer."
- "This tumor marker proves cancer is back."
- "This supplement is safe with chemotherapy."

Escalation/review boundary
If a request asks for diagnosis, prognosis, treatment changes, genetic-risk prediction, or tumor-marker conclusions, refuse the claim and route to clinician or genetics-trained review. If urgent symptoms are present, prioritize urgent clinical contact over education.
