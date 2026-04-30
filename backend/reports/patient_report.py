def build_patient_report(
    patient_state,
    labs,
    trends,
    risks,
    treatment_effects,
    radiology_summary,
    symptoms,
    ai_summary,
):
    return {
        "patient_state": patient_state,
        "latest_labs": labs.iloc[-1].to_dict(),
        "baseline_labs": labs.iloc[0].to_dict(),
        "lab_history": labs.to_dict(orient="records"),
        "trends": trends,
        "risks": risks,
        "treatment_effects": treatment_effects,
        "radiology_summary": radiology_summary,
        "breast_imaging_summary": radiology_summary,
        "symptoms": symptoms.to_dict(orient="records") if symptoms is not None and not symptoms.empty else [],
        "ai_summary": ai_summary,
        "safety_note": "Breast cancer clinical decision-support only. Not for diagnosis, cancer detection, confirming metastasis, or replacing a licensed clinician.",
    }
