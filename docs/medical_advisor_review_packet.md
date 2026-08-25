# Unreviewed Clinical Advisor Packet

Generated: 2026-08-25T08:14:04.714518+00:00

Status: prepared for future clinician review; not reviewed or approved.

## Claim Boundary
This unreviewed packet is prepared for future expert review of an engineering prototype. It has not been reviewed or approved by a clinician and is not clinical sign-off, clinical validation, or authorization for patient care use.

## Review Scope
- urgent symptom/red-flag vocabulary
- patient-reported severity to CTCAE-style review hints
- CBC population-default reference context and limitations
- supplement/medication interaction review-routing rules
- genetics/VUS/tumor-marker refusal and education boundaries
- minimum evidence standards for model and RAG outputs

## Urgent Symptom Terms
blood discharge, chest pain, confusion, difficulty breathing, fainting, fever, severe headache, shortness of breath, uncontrolled bleeding

## Genetics Boundary
Genetic Counseling Readiness organizes information for oncology/genetics review. It does not diagnose inherited risk, interpret variants as medical advice, or recommend treatment changes.

## Expanded Family-History Fields
- bilateral_breast_cancer
- multiple_primary_cancers
- ancestry_ethnicity
- prior_breast_biopsy_atypia
- relation_degree

## Supplement/Medication Interaction Rules
- st_johns_wort_cyp_interaction: St. John's wort can affect drug-metabolism pathways and may interact with oncology medicines. Do not start or stop it without oncology-team/pharmacist review.
- grapefruit_cdk46_interaction: Grapefruit products can interact with several oral cancer medicines. Ask the oncology team or pharmacist before using them with CDK4/6 inhibitors.
- bleeding_risk_supplements: Some supplements may affect bleeding risk or procedures. This system cannot determine safety; review with the care team/pharmacist before use.
- cbd_cyp_interaction: CBD/cannabis products may interact with medicines or increase sedation/nausea effects. Discuss use with the oncology team/pharmacist first.
- antioxidants_during_treatment: High-dose antioxidant supplements during chemotherapy or radiation should be reviewed with the oncology team. Do not use them as a replacement for prescribed treatment.

## Questions For Advisor
- Which urgent symptom terms are missing or too broad?
- Are CBC reference-context warnings phrased safely for patients?
- Which supplement interaction rules should be added, removed, or clinician-only?
- Are genetics/VUS/tumor-marker boundaries conservative enough?
- Which family-history fields are required before genetic-counseling readiness is credible?
