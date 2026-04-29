import pandas as pd

# Load data
labs = pd.read_csv("data/labs.csv", parse_dates=["date"])
treatment = pd.read_csv("data/treatment.csv", parse_dates=["date"])

# Sort by time (VERY important for longitudinal data)
labs = labs.sort_values("date")
treatment = treatment.sort_values("date")

print("Labs:")
print(labs)

print("\nTreatment:")
print(treatment)

from backend.processing.trend_analysis import analyze_labs

trends = analyze_labs(labs)

print("\nTrends:")
print(trends)

from backend.processing.risk_engine import detect_risks, detect_trend_risk

risks = detect_risks(labs)

print("\nRisks:")
print(risks)

from backend.processing.treatment_analysis import align_labs_with_treatment

treatment_effects = align_labs_with_treatment(labs, treatment)

print("\nTreatment Effects:")
print(treatment_effects)

trend_risks = detect_trend_risk(labs)

print("\nTrend Risks:")
print(trend_risks)

# --- AI Clinical Summary ---
from backend.processing.clinical_summary import generate_clinical_summary

all_risks = risks + trend_risks
summary = generate_clinical_summary(trends, all_risks, treatment_effects)

print("\n" + "=" * 60)
print("AI CLINICAL SUMMARY")
print("=" * 60)
print(summary)