def build_patient_report(
    patient_state,
    labs,
    trends,
    risks,
    treatment_effects,
    radiology_summary,
    symptoms,
    timeline,
    ai_summary,
):
    latest_labs = labs.iloc[-1].to_dict() if labs is not None and not labs.empty else None
    baseline_labs = labs.iloc[0].to_dict() if labs is not None and not labs.empty else None
    lab_history = labs.to_dict(orient="records") if labs is not None and not labs.empty else []

    return {
        "patient_state": patient_state,
        "latest_labs": latest_labs,
        "baseline_labs": baseline_labs,
        "lab_history": lab_history,
        "trends": trends,
        "risks": risks,
        "treatment_effects": treatment_effects,
        "radiology_summary": radiology_summary,
        "breast_imaging_summary": radiology_summary,
        "symptoms": symptoms.to_dict(orient="records") if symptoms is not None and not symptoms.empty else [],
        "timeline": timeline,
        "ai_summary": ai_summary,
        "safety_note": "Breast cancer clinical decision-support only. Not for diagnosis, cancer detection, confirming metastasis, or replacing a licensed clinician.",
    }
