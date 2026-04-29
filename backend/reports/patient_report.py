def build_patient_report(labs, trends, risks, trend_risks, treatment_effects, ai_summary):
    return {
        "patient_id": "P001",
        "latest_labs": labs.iloc[-1].to_dict(),
        "baseline_labs": labs.iloc[0].to_dict(),
        "trends": trends,
        "risks": risks + trend_risks,
        "treatment_effects": treatment_effects,
        "ai_summary": ai_summary
    }