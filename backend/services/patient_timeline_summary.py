def build_patient_timeline_risk_summary(report):
    assessment = report.get("multimodal_assessment") or {}
    signals = assessment.get("signals") or {}
    mri_signal = signals.get("mri_response") or {}
    clinical_signal = signals.get("clinical_monitoring") or {}
    symptom_signal = signals.get("symptoms") or {}

    latest_labs = report.get("latest_labs") or {}
    treatments = report.get("treatment_effects") or []
    timeline = report.get("timeline") or []
    risks = report.get("risks") or []
    outcome = report.get("treatment_outcome") or {}
    synthetic_prediction = report.get("synthetic_model_prediction") or {}

    score = assessment.get("treatment_monitoring_score")
    overall_status = assessment.get("overall_status") or "unknown"
    response_probability = _response_probability(synthetic_prediction, mri_signal)

    return {
        "headline": _headline(overall_status, score),
        "patient_summary": _patient_summary(
            overall_status=overall_status,
            score=score,
            mri_signal=mri_signal,
            clinical_signal=clinical_signal,
            symptom_signal=symptom_signal,
            response_probability=response_probability,
        ),
        "key_points": _key_points(
            latest_labs=latest_labs,
            treatments=treatments,
            mri_signal=mri_signal,
            response_probability=response_probability,
            outcome=outcome,
        ),
        "review_flags": _review_flags(risks, assessment),
        "uncertainty_notes": _uncertainty_notes(report, mri_signal, synthetic_prediction),
        "clinician_summary": _clinician_summary(
            overall_status=overall_status,
            clinical_signal=clinical_signal,
            symptom_signal=symptom_signal,
            risks=risks,
            timeline=timeline,
        ),
        "generated_by": "deterministic_patient_timeline_risk_summary_engine",
        "safety_note": (
            "This is a monitoring summary for discussion with the care team. "
            "It does not diagnose, confirm treatment response, or choose treatment."
        ),
    }


def _headline(overall_status, score):
    if overall_status == "needs_clinician_review":
        return f"Review recommended based on monitoring signals{_score_text(score)}."
    if overall_status == "watch_closely":
        return f"Mixed signals; continue close monitoring{_score_text(score)}."
    if overall_status == "favorable_response_signal":
        return f"Available signals look favorable{_score_text(score)}."
    return f"No major combined warning pattern is shown{_score_text(score)}."


def _score_text(score):
    return f" (score {score}/100)" if score is not None else ""


def _patient_summary(overall_status, score, mri_signal, clinical_signal, symptom_signal, response_probability):
    response_text = mri_signal.get("message") or "No imaging/model response signal is available yet."
    clinical_text = clinical_signal.get("message") or "No CBC risk signal is available yet."
    symptom_text = symptom_signal.get("message") or "No symptom signal is available yet."
    probability_text = (
        f" The demo model response probability is {round(response_probability * 100, 1)}%."
        if response_probability is not None else ""
    )

    if overall_status == "needs_clinician_review":
        opener = "The portal sees some items that should be reviewed with the oncology team."
    elif overall_status == "watch_closely":
        opener = "The portal sees a mixed picture, so the safest interpretation is continued monitoring."
    else:
        opener = "The portal does not see a major combined warning pattern in the available data."

    return f"{opener} {response_text} {clinical_text} {symptom_text}{probability_text}"


def _key_points(latest_labs, treatments, mri_signal, response_probability, outcome):
    points = []

    if response_probability is not None:
        points.append(f"Demo response model probability: {round(response_probability * 100, 1)}%.")
    elif mri_signal.get("message"):
        points.append(mri_signal["message"])

    if latest_labs:
        points.append(
            "Latest CBC: "
            f"WBC {latest_labs.get('wbc', 'n/a')}, "
            f"hemoglobin {latest_labs.get('hemoglobin', 'n/a')}, "
            f"platelets {latest_labs.get('platelets', 'n/a')}."
        )

    if treatments:
        latest_cycle = treatments[-1]
        points.append(
            f"Latest treatment-cycle window tracked: cycle {latest_cycle.get('cycle')} "
            f"({latest_cycle.get('drug')})."
        )

    if outcome:
        points.append(
            f"Recorded synthetic outcome: {outcome.get('response_category')} / "
            f"{outcome.get('cancer_status')}."
        )

    return points[:5]


def _review_flags(risks, assessment):
    flags = [risk.get("message") for risk in risks if risk.get("message")]
    recommended_action = assessment.get("recommended_action")
    if recommended_action:
        flags.insert(0, recommended_action)
    return _dedupe(flags)[:5]


def _uncertainty_notes(report, mri_signal, synthetic_prediction):
    notes = []
    if report.get("has_synthetic_labs"):
        notes.append("CBC rows are synthetic demo data in this project build.")
    if mri_signal.get("source") == "complete_synthetic_longitudinal_model":
        notes.append("The displayed response model is trained on synthetic longitudinal data, not clinically validated outcomes.")
    if not synthetic_prediction and mri_signal.get("source") == "none":
        notes.append("No model prediction is available for this patient.")
    if not report.get("symptoms"):
        notes.append("Symptom tracking is incomplete or unavailable.")
    if not report.get("lab_history"):
        notes.append("CBC trend tracking is incomplete or unavailable.")
    if not report.get("timeline"):
        notes.append("Timeline context is incomplete.")
    return notes[:5]


def _clinician_summary(overall_status, clinical_signal, symptom_signal, risks, timeline):
    urgent_count = clinical_signal.get("urgent_count", 0)
    watch_count = clinical_signal.get("watch_count", 0)
    max_symptom = symptom_signal.get("max_severity")
    recent_events = len(timeline[-14:]) if timeline else 0
    risk_types = ", ".join(_dedupe([risk.get("type") for risk in risks if risk.get("type")])[:5])
    if not risk_types:
        risk_types = "none listed"

    symptom_part = f"max symptom severity {max_symptom}/10" if max_symptom is not None else "no symptom severity available"
    return (
        f"Status: {overall_status}. CBC/risk flags: {urgent_count} urgent, {watch_count} watch. "
        f"Symptoms: {symptom_part}. Recent timeline events represented: {recent_events}. "
        f"Risk types: {risk_types}."
    )


def _response_probability(synthetic_prediction, mri_signal):
    for key in [
        "logistic_regression_probability",
        "gradient_boosting_probability",
        "random_forest_probability",
        "extra_trees_probability",
        "temporal_1d_cnn_probability",
        "temporal_gru_probability",
        "response_probability",
        "pcr_probability",
    ]:
        value = synthetic_prediction.get(key) if key in synthetic_prediction else mri_signal.get(key)
        if value is not None:
            return float(value)
    return None


def _dedupe(items):
    seen = set()
    output = []
    for item in items:
        if not item or item in seen:
            continue
        output.append(item)
        seen.add(item)
    return output
