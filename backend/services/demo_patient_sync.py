from datetime import date

from backend.models import (
    BreastCancerProfile,
    ClinicalIntervention,
    CTReport,
    ImagingReport,
    LabResult,
    MedicationLog,
    Patient,
    SymptomReport,
    Treatment,
    TreatmentOutcome,
)


DEMO_PATIENT_ID = "P001"
DEMO_SOURCE = "curated_synthetic_demo_journey"


def sync_demo_patient_journey(db, patient_id=DEMO_PATIENT_ID):
    """Reset the demo patient to a coherent synthetic treatment journey."""

    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if patient is None:
        patient = Patient(
            id=patient_id,
            name=f"Patient {patient_id}",
            diagnosis="Breast cancer - doctor-confirmed",
        )
        db.add(patient)
    else:
        patient.name = f"Patient {patient_id}"
        patient.diagnosis = "Breast cancer - doctor-confirmed"

    profile = db.query(BreastCancerProfile).filter(BreastCancerProfile.patient_id == patient_id).first()
    if profile is None:
        profile = BreastCancerProfile(patient_id=patient_id)
        db.add(profile)
    profile.cancer_stage = "Stage II"
    profile.er_status = "positive"
    profile.pr_status = "positive"
    profile.her2_status = "negative"
    profile.molecular_subtype = "HR-positive / HER2-negative"
    profile.treatment_intent = "neoadjuvant chemotherapy monitoring"
    profile.menopausal_status = "premenopausal"

    for model in [
        CTReport,
        ImagingReport,
        Treatment,
        LabResult,
        SymptomReport,
        MedicationLog,
        ClinicalIntervention,
        TreatmentOutcome,
    ]:
        db.query(model).filter(model.patient_id == patient_id).delete(synchronize_session=False)

    treatments = [
        Treatment(patient_id=patient_id, date=date(2026, 1, 5), cycle=1, drug="Dose-dense AC (doxorubicin/cyclophosphamide)"),
        Treatment(patient_id=patient_id, date=date(2026, 1, 19), cycle=2, drug="Dose-dense AC (doxorubicin/cyclophosphamide)"),
        Treatment(patient_id=patient_id, date=date(2026, 2, 2), cycle=3, drug="Dose-dense AC (doxorubicin/cyclophosphamide)"),
        Treatment(patient_id=patient_id, date=date(2026, 2, 16), cycle=4, drug="Dose-dense AC (doxorubicin/cyclophosphamide)"),
        Treatment(patient_id=patient_id, date=date(2026, 3, 2), cycle=5, drug="Paclitaxel"),
        Treatment(patient_id=patient_id, date=date(2026, 3, 16), cycle=6, drug="Paclitaxel"),
    ]

    lab_rows = [
        (date(2026, 1, 2), 6.8, 13.1, 252, "Baseline CBC before neoadjuvant treatment."),
        (date(2026, 1, 4), 6.5, 13.0, 248, "Pre-cycle 1 CBC."),
        (date(2026, 1, 14), 3.1, 12.5, 190, "Cycle 1 expected nadir CBC."),
        (date(2026, 1, 18), 4.8, 12.7, 224, "Recovery CBC before cycle 2."),
        (date(2026, 1, 28), 2.4, 12.0, 165, "Cycle 2 nadir CBC."),
        (date(2026, 2, 1), 4.0, 12.1, 205, "Recovery CBC before cycle 3."),
        (date(2026, 2, 11), 2.1, 11.6, 150, "Cycle 3 nadir CBC; clinician review threshold watch."),
        (date(2026, 2, 15), 3.7, 11.8, 188, "Recovery CBC before cycle 4."),
        (date(2026, 2, 25), 2.6, 11.5, 158, "Cycle 4 nadir CBC."),
        (date(2026, 3, 1), 4.1, 11.7, 196, "Recovery CBC before paclitaxel."),
        (date(2026, 3, 11), 3.2, 11.2, 170, "Cycle 5 nadir CBC."),
        (date(2026, 3, 15), 4.6, 11.4, 205, "Recovery CBC before cycle 6."),
        (date(2026, 3, 25), 3.4, 10.9, 178, "Cycle 6 nadir CBC."),
        (date(2026, 3, 29), 4.7, 11.1, 214, "End-of-course recovery CBC."),
    ]
    labs = [
        LabResult(
            patient_id=patient_id,
            date=lab_date,
            wbc=wbc,
            hemoglobin=hgb,
            platelets=platelets,
            source=DEMO_SOURCE,
            source_note=f"Synthetic demo only. {note}",
        )
        for lab_date, wbc, hgb, platelets, note in lab_rows
    ]

    symptoms = [
        SymptomReport(patient_id=patient_id, date=date(2026, 1, 12), symptom="nausea", severity=5, notes="Mild nausea after cycle 1; synthetic demo data."),
        SymptomReport(patient_id=patient_id, date=date(2026, 1, 30), symptom="fatigue", severity=6, notes="More tired during cycle 2 nadir window; synthetic demo data."),
        SymptomReport(patient_id=patient_id, date=date(2026, 2, 12), symptom="mouth sores", severity=7, notes="Painful eating after cycle 3; synthetic demo data."),
        SymptomReport(patient_id=patient_id, date=date(2026, 3, 10), symptom="neuropathy", severity=4, notes="Mild tingling after paclitaxel cycle 5; synthetic demo data."),
        SymptomReport(patient_id=patient_id, date=date(2026, 3, 24), symptom="fatigue", severity=5, notes="Moderate fatigue after cycle 6; synthetic demo data."),
    ]

    imaging_reports = [
        ImagingReport(
            patient_id=patient_id,
            date=date(2026, 1, 3),
            modality="Breast MRI",
            report_type="Baseline breast MRI",
            body_site="Breast",
            findings="Right upper outer breast enhancing mass measures 4.2 cm. Prominent right axillary node. BI-RADS 6.",
            impression="Known biopsy-proven right breast malignancy. Baseline MRI before neoadjuvant chemotherapy.",
        ),
        ImagingReport(
            patient_id=patient_id,
            date=date(2026, 2, 28),
            modality="Breast MRI",
            report_type="Interim response MRI after AC cycle 4",
            body_site="Breast",
            findings="Right breast mass measures 2.7 cm, decreased from 4.2 cm. Axillary node appears smaller. No metastatic wording identified in this synthetic report.",
            impression="Interval decrease in right breast mass size after four AC cycles. Findings require clinician interpretation.",
        ),
        ImagingReport(
            patient_id=patient_id,
            date=date(2026, 3, 30),
            modality="Breast MRI",
            report_type="End-of-course response MRI",
            body_site="Breast",
            findings="Residual right breast enhancement measures 1.8 cm. No new suspicious breast lesion described in this synthetic report.",
            impression="Further interval decrease in right breast mass size compared with baseline and interim MRI. Not a diagnosis of complete response.",
        ),
    ]

    medications = [
        MedicationLog(patient_id=patient_id, date=date(2026, 1, 5), medication="ondansetron", dose="8 mg", frequency="as needed", notes="Synthetic anti-nausea support medication.", source=DEMO_SOURCE),
        MedicationLog(patient_id=patient_id, date=date(2026, 1, 6), medication="pegfilgrastim", dose="6 mg", frequency="once after AC cycle", notes="Synthetic growth-factor support after dose-dense AC.", source=DEMO_SOURCE),
        MedicationLog(patient_id=patient_id, date=date(2026, 2, 3), medication="pegfilgrastim", dose="6 mg", frequency="once after AC cycle", notes="Synthetic growth-factor support after cycle 3.", source=DEMO_SOURCE),
    ]

    interventions = [
        ClinicalIntervention(
            patient_id=patient_id,
            date=date(2026, 2, 12),
            intervention_type="symptom_review_recommended",
            reason="Mouth sores severity 7/10 during cycle 3 nadir window.",
            medication_or_product=None,
            dose=None,
            notes="Synthetic clinician-review cue; not a treatment order.",
            source=DEMO_SOURCE,
        ),
    ]

    outcome = TreatmentOutcome(
        patient_id=patient_id,
        assessment_date=date(2026, 4, 10),
        response_category="partial imaging response signal",
        cancer_status="monitoring signal only - clinician interpretation required",
        maintenance_plan="Discuss MRI, CBC recovery, symptoms, and next-step planning with oncology team.",
        recurrence_risk_band="not assessed in PoC",
        notes="Synthetic outcome-style summary for UI workflow only; not clinical validation.",
        source=DEMO_SOURCE,
    )

    db.add_all(treatments + labs + symptoms + imaging_reports + medications + interventions + [outcome])
    db.commit()

    return {
        "patient_id": patient_id,
        "treatments": len(treatments),
        "labs": len(labs),
        "symptoms": len(symptoms),
        "imaging_reports": len(imaging_reports),
        "medications": len(medications),
        "interventions": len(interventions),
        "source": DEMO_SOURCE,
    }
