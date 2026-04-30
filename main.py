import json
import pandas as pd

# --- Module Imports ---
from backend.processing.trend_analysis import analyze_labs
from backend.processing.risk_engine import detect_risks, detect_trend_risk
from backend.processing.treatment_analysis import align_labs_with_treatment
from backend.processing.radiology_analysis import analyze_radiology_reports
from backend.processing.clinical_summary import generate_clinical_summary
from backend.processing.patient_state import build_patient_state
from backend.reports.patient_report import build_patient_report

# --- 1. Load Data ---
labs = pd.read_csv("data/labs.csv", parse_dates=["date"])
treatment = pd.read_csv("data/treatment.csv", parse_dates=["date"])
ct_reports = pd.read_csv("Data/ct_reports.csv", parse_dates=["date"])

# Sort by time (VERY important for longitudinal data)
labs = labs.sort_values("date")
treatment = treatment.sort_values("date")

print("Labs:")
print(labs)
print("\nTreatment:")
print(treatment)

# --- 2. Process Data ---
trends = analyze_labs(labs)
risks = detect_risks(labs)
trend_risks = detect_trend_risk(labs)
treatment_effects = align_labs_with_treatment(labs, treatment)
radiology_summary = analyze_radiology_reports(ct_reports)

print("\nTrends:")
print(trends)
print("\nRisks:")
print(risks)
print("\nTreatment Effects:")
print(treatment_effects)
print("\nTrend Risks:")
print(trend_risks)
print("\nRadiology Summary:")
print(radiology_summary)

# --- 3. AI Clinical Summary ---
all_risks = risks + trend_risks
class DemoPatient:
    id = "P001"
    name = "Patient P001"
    diagnosis = "Breast cancer - doctor-confirmed"


patient_state = build_patient_state(
    patient=DemoPatient(),
    breast_profile=None,
    labs=labs,
    trends=trends,
    risks=all_risks,
    treatment_effects=treatment_effects,
    radiology_summary=radiology_summary,
)
summary = generate_clinical_summary(patient_state)

print("\n" + "=" * 60)
print("AI CLINICAL SUMMARY")
print("=" * 60)
print(summary)

# --- 4. Generate & Save Patient Report ---
report = build_patient_report(
    patient_state=patient_state,
    labs=labs,
    trends=trends,
    risks=all_risks,
    treatment_effects=treatment_effects,
    radiology_summary=radiology_summary,
    symptoms=None,
    ai_summary=summary
)

output_file = "Data/patient_report.json"
with open(output_file, "w") as f:
    json.dump(report, f, indent=4, default=str)

print(f"\nPatient report saved to {output_file}")
