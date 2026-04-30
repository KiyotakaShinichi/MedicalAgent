def build_patient_state(
    patient,
    labs,
    trends,
    risks,
    treatment_effects,
    radiology_summary,
    symptoms=None,
):
    latest_labs = labs.iloc[-1].to_dict() if not labs.empty else None
    baseline_labs = labs.iloc[0].to_dict() if not labs.empty else None

    state = {
        "patient": {
            "id": patient.id,
            "name": patient.name,
            "diagnosis": patient.diagnosis,
        },
        "latest_labs": latest_labs,
        "baseline_labs": baseline_labs,
        "lab_trends": trends,
        "treatment_effects": treatment_effects,
        "radiology": radiology_summary,
        "symptoms": symptoms.to_dict(orient="records") if symptoms is not None and not symptoms.empty else [],
        "risk_flags": risks,
        "safety_positioning": {
            "role": "clinical decision-support and longitudinal monitoring assistant",
            "not_for": ["diagnosis", "cancer detection", "replacing clinician judgment"],
            "requires_clinician_review": True,
        },
    }

    return state
