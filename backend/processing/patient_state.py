def build_patient_state(
    patient,
    breast_profile,
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
        "breast_cancer_profile": _profile_to_dict(breast_profile),
        "latest_labs": latest_labs,
        "baseline_labs": baseline_labs,
        "lab_trends": trends,
        "treatment_effects": treatment_effects,
        "radiology": radiology_summary,
        "symptoms": symptoms.to_dict(orient="records") if symptoms is not None and not symptoms.empty else [],
        "risk_flags": risks,
        "safety_positioning": {
            "role": "breast cancer clinical decision-support and longitudinal monitoring assistant",
            "not_for": ["diagnosis", "cancer detection", "confirming metastasis", "replacing clinician judgment"],
            "requires_clinician_review": True,
        },
    }

    return state


def _profile_to_dict(profile):
    if profile is None:
        return None

    return {
        "cancer_stage": profile.cancer_stage,
        "er_status": profile.er_status,
        "pr_status": profile.pr_status,
        "her2_status": profile.her2_status,
        "molecular_subtype": profile.molecular_subtype,
        "treatment_intent": profile.treatment_intent,
        "menopausal_status": profile.menopausal_status,
    }
