def build_data_availability(report):
    labs = report.get("lab_history") or []
    symptoms = report.get("symptoms") or []
    timeline = report.get("timeline") or []
    treatment_effects = report.get("treatment_effects") or []
    radiology_summary = report.get("radiology_summary")
    mri_registry = report.get("mri_registry") or []
    synthetic_prediction = report.get("synthetic_model_prediction")
    assessment = report.get("multimodal_assessment") or {}
    mri_signal = (assessment.get("signals") or {}).get("mri_response") or {}

    items = [
        _item(
            "CBC trend",
            "available" if len(labs) >= 2 else "insufficient_data" if labs else "missing",
            f"{len(labs)} CBC row(s) available.",
            "CBC trends need at least two timepoints to compare direction.",
        ),
        _item(
            "Treatment-cycle alignment",
            "available" if treatment_effects else "insufficient_data",
            f"{len(treatment_effects)} aligned treatment window(s).",
            "Add treatment dates and CBC rows around each cycle to interpret toxicity over time.",
        ),
        _item(
            "Symptoms",
            "available" if symptoms else "missing",
            f"{len(symptoms)} symptom report(s) available.",
            "Symptoms are optional, but missing symptoms reduce confidence in toxicity summaries.",
        ),
        _item(
            "Imaging trend",
            "available" if radiology_summary or mri_registry else "missing",
            "Imaging report NLP or MRI file references are present." if radiology_summary or mri_registry else "No imaging trend source is present.",
            "Add MRI report text or MRI file references to support response trend monitoring.",
        ),
        _item(
            "Model signal",
            _model_status(mri_signal, synthetic_prediction),
            _model_detail(mri_signal, synthetic_prediction),
            "A missing model signal should fall back to report NLP and clinician review.",
        ),
        _item(
            "Timeline depth",
            "available" if len(timeline) >= 4 else "insufficient_data" if timeline else "missing",
            f"{len(timeline)} timeline event(s) represented.",
            "Longitudinal monitoring needs repeated labs, treatments, symptoms, and imaging over time.",
        ),
    ]
    status = _worst_status(item["status"] for item in items)
    return {
        "status": status,
        "items": items,
        "clinician_style_summary": _clinician_style_summary(status, items),
        "patient_friendly_summary": _patient_summary(status, items),
        "fallback_policy": (
            "When data are missing, invalid, or model signals are unavailable, the system should explain the limitation "
            "and route the patient summary toward clinician review instead of forcing a prediction."
        ),
    }


def _model_status(mri_signal, synthetic_prediction):
    if synthetic_prediction:
        return "available"
    if mri_signal.get("status") and mri_signal.get("status") != "unavailable":
        return "available"
    if mri_signal.get("status") == "unavailable":
        return "model_unavailable"
    return "model_unavailable"


def _model_detail(mri_signal, synthetic_prediction):
    if synthetic_prediction:
        return "Synthetic longitudinal model prediction is available."
    source = mri_signal.get("source")
    if source and source != "none":
        return f"Response signal is available from {source}."
    return "No registered or patient-specific model signal is available."


def _item(name, status, detail, next_step):
    return {
        "name": name,
        "status": status,
        "detail": detail,
        "next_step": next_step,
    }


def _clinician_style_summary(status, items):
    missing = [item["name"] for item in items if item["status"] in {"missing", "insufficient_data", "model_unavailable"}]
    if not missing:
        return "Data availability is sufficient for longitudinal monitoring summary generation."
    return (
        "Interpret with limitations: "
        + ", ".join(missing[:5])
        + " are missing, insufficient, or unavailable. Route summary for clinician review if clinical concern persists."
    )


def _patient_summary(status, items):
    if status == "available":
        return "Your record has enough repeated information for the portal to summarize trends."
    return (
        "Some parts of your record are incomplete, so the portal may be less confident. "
        "Your care team can still review the information you have entered."
    )


def _worst_status(statuses):
    order = {
        "available": 0,
        "insufficient_data": 1,
        "missing": 2,
        "model_unavailable": 2,
    }
    worst = "available"
    for status in statuses:
        if order.get(status, 3) > order.get(worst, 0):
            worst = status
    return worst
