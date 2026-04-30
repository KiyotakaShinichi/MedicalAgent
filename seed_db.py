from datetime import date

from backend.database import SessionLocal
from backend.models import (
    BreastCancerProfile,
    CTReport,
    ImagingReport,
    LabResult,
    MRIFileRegistry,
    Patient,
    SymptomReport,
    Treatment,
)
from backend.schema_migrations import ensure_schema


ensure_schema()
db = SessionLocal()

# Clear old demo data
db.query(CTReport).delete()
db.query(MRIFileRegistry).delete()
db.query(ImagingReport).delete()
db.query(Treatment).delete()
db.query(LabResult).delete()
db.query(SymptomReport).delete()
db.query(BreastCancerProfile).delete()
db.query(Patient).delete()

patient = Patient(
    id="P001",
    name="Patient P001",
    diagnosis="Breast cancer - doctor-confirmed",
)

profile = BreastCancerProfile(
    patient_id="P001",
    cancer_stage="Stage II",
    er_status="positive",
    pr_status="positive",
    her2_status="negative",
    molecular_subtype="HR-positive / HER2-negative",
    treatment_intent="neoadjuvant chemotherapy monitoring",
    menopausal_status="premenopausal",
)

labs = [
    LabResult(patient_id="P001", date=date(2026, 1, 1), wbc=6.5, hemoglobin=13.2, platelets=250),
    LabResult(patient_id="P001", date=date(2026, 1, 8), wbc=4.2, hemoglobin=12.5, platelets=210),
    LabResult(patient_id="P001", date=date(2026, 1, 15), wbc=2.1, hemoglobin=11.8, platelets=150),
    LabResult(patient_id="P001", date=date(2026, 1, 22), wbc=3.8, hemoglobin=12.0, platelets=180),
]

treatments = [
    Treatment(patient_id="P001", date=date(2026, 1, 1), cycle=1, drug="Doxorubicin/Cyclophosphamide"),
    Treatment(patient_id="P001", date=date(2026, 1, 15), cycle=2, drug="Doxorubicin/Cyclophosphamide"),
]

symptoms = [
    SymptomReport(patient_id="P001", date=date(2026, 1, 10), symptom="fatigue", severity=5, notes="More tired after cycle 1."),
    SymptomReport(patient_id="P001", date=date(2026, 1, 18), symptom="mouth sores", severity=7, notes="Painful eating after cycle 2."),
]

imaging_reports = [
    ImagingReport(
        patient_id="P001",
        date=date(2026, 1, 1),
        modality="Breast MRI",
        report_type="Baseline breast MRI",
        body_site="Breast",
        findings="Right breast upper outer quadrant enhancing mass measuring 4.2 cm. Prominent right axillary lymph node noted. BI-RADS 6.",
        impression="Known biopsy-proven right breast malignancy. Baseline MRI before neoadjuvant chemotherapy.",
    ),
    ImagingReport(
        patient_id="P001",
        date=date(2026, 2, 1),
        modality="Breast MRI",
        report_type="Follow-up breast MRI",
        body_site="Breast",
        findings="Right breast upper outer quadrant enhancing mass measuring 3.1 cm. Right axillary lymph node slightly decreased. No evidence of liver, lung, or bone metastatic lesion on available staging report text.",
        impression="Interval decrease in breast tumor size compared with baseline MRI.",
    ),
]

db.add(patient)
db.add(profile)
db.add_all(labs)
db.add_all(treatments)
db.add_all(symptoms)
db.add_all(imaging_reports)

db.commit()
db.close()

print("Seed data inserted.")
