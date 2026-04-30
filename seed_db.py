from datetime import date

from backend.database import SessionLocal
from backend.models import CTReport, LabResult, Patient, SymptomReport, Treatment


db = SessionLocal()

# Clear old demo data
db.query(CTReport).delete()
db.query(Treatment).delete()
db.query(LabResult).delete()
db.query(SymptomReport).delete()
db.query(Patient).delete()

patient = Patient(
    id="P001",
    name="Patient P001",
    diagnosis="Lung cancer - doctor-confirmed",
)

labs = [
    LabResult(patient_id="P001", date=date(2026, 1, 1), wbc=6.5, hemoglobin=13.2, platelets=250),
    LabResult(patient_id="P001", date=date(2026, 1, 8), wbc=4.2, hemoglobin=12.5, platelets=210),
    LabResult(patient_id="P001", date=date(2026, 1, 15), wbc=2.1, hemoglobin=11.8, platelets=150),
    LabResult(patient_id="P001", date=date(2026, 1, 22), wbc=3.8, hemoglobin=12.0, platelets=180),
]

treatments = [
    Treatment(patient_id="P001", date=date(2026, 1, 1), cycle=1, drug="Cisplatin"),
    Treatment(patient_id="P001", date=date(2026, 1, 15), cycle=2, drug="Cisplatin"),
]

symptoms = [
    SymptomReport(patient_id="P001", date=date(2026, 1, 10), symptom="fatigue", severity=5, notes="More tired after cycle 1."),
    SymptomReport(patient_id="P001", date=date(2026, 1, 18), symptom="shortness of breath", severity=7, notes="Occurs with walking upstairs."),
]

ct_reports = [
    CTReport(
        patient_id="P001",
        date=date(2026, 1, 1),
        report_type="Baseline CT",
        findings="Right upper lobe mass measuring 4.2 cm. Mediastinal lymph nodes noted.",
        impression="Known lung malignancy. Baseline scan.",
    ),
    CTReport(
        patient_id="P001",
        date=date(2026, 2, 1),
        report_type="Follow-up CT",
        findings="Right upper lobe mass measuring 3.5 cm. Mediastinal lymph nodes slightly decreased. No new liver, adrenal, or bone lesion described.",
        impression="Interval decrease in tumor size compared with baseline.",
    ),
]

db.add(patient)
db.add_all(labs)
db.add_all(treatments)
db.add_all(symptoms)
db.add_all(ct_reports)

db.commit()
db.close()

print("Seed data inserted.")
