from datetime import datetime, timedelta


def build_timeline_intelligence(report):
    timeline = report.get("timeline") or []
    labs = report.get("lab_history") or []
    symptoms = report.get("symptoms") or []
    assessment = report.get("multimodal_assessment") or {}
    risks = report.get("risks") or []
    mri_signal = (assessment.get("signals") or {}).get("mri_response") or {}
    clinical_signal = (assessment.get("signals") or {}).get("clinical_monitoring") or {}

    return {
        "toxicity_trend": _toxicity_trend(labs, risks),
        "response_trend": _response_trend(mri_signal, report),
        "discordance": _discordance(mri_signal, clinical_signal),
        "last_14_days": _last_n_days(timeline, 14),
        "missing_data_warnings": _missing_data_warnings(report),
        "tumor_board_brief": _tumor_board_brief(report, assessment),
        "supported_questions": [
            "What changed in the last 14 days?",
            "Has toxicity increased since cycle 2?",
            "Is imaging response improving while CBC toxicity worsens?",
            "Summarize this patient for tumor board review.",
        ],
    }


def answer_timeline_question(report, question):
    lower = question.lower()
    intelligence = build_timeline_intelligence(report)

    if "last 14" in lower or "last fourteen" in lower or "what changed" in lower:
        events = intelligence["last_14_days"]["events"]
        if not events:
            return _answer("No timeline events were found in the last 14 days represented in the record.", intelligence)
        summaries = [f"{event.get('date')}: {event.get('title')} - {event.get('summary')}" for event in events[:8]]
        return _answer("Recent timeline changes: " + " | ".join(summaries), intelligence)

    if "toxicity" in lower or "cycle 2" in lower or "cbc" in lower:
        trend = intelligence["toxicity_trend"]
        return _answer(
            f"CBC toxicity trend is {trend['status']}. {trend['message']}",
            intelligence,
        )

    if "imaging" in lower or "mri" in lower or "response" in lower:
        response = intelligence["response_trend"]
        discordance = intelligence["discordance"]
        return _answer(
            f"Response trend: {response['message']} Discordance check: {discordance['message']}",
            intelligence,
        )

    if "tumor board" in lower or "doctor" in lower or "clinician" in lower or "summarize" in lower:
        return _answer(intelligence["tumor_board_brief"], intelligence)

    return _answer(
        "I can summarize recent changes, CBC toxicity trend, MRI/response trend, discordance, or tumor-board style brief.",
        intelligence,
    )


def _answer(text, intelligence):
    return {
        "answer": text,
        "intelligence": intelligence,
        "safety_note": "Timeline intelligence is monitoring support only and needs clinician interpretation.",
    }


def _toxicity_trend(labs, risks):
    if not labs:
        return {"status": "unknown", "message": "No CBC data is available."}

    sorted_labs = sorted(labs, key=lambda row: str(row.get("date")))
    midpoint = max(1, len(sorted_labs) // 2)
    early = sorted_labs[:midpoint]
    late = sorted_labs[midpoint:] or sorted_labs[-1:]

    early_min_wbc = min(float(row.get("wbc", 999)) for row in early)
    late_min_wbc = min(float(row.get("wbc", 999)) for row in late)
    early_min_platelets = min(float(row.get("platelets", 9999)) for row in early)
    late_min_platelets = min(float(row.get("platelets", 9999)) for row in late)
    urgent_cbc = [risk for risk in risks if risk.get("category") in {"lab", "lab_trend", "deterministic_clinical_rule"} and risk.get("severity") == "urgent_review"]

    if late_min_wbc < early_min_wbc * 0.8 or late_min_platelets < early_min_platelets * 0.8:
        status = "worsening"
        message = "Later cycles show lower CBC nadirs than earlier cycles."
    elif urgent_cbc:
        status = "clinician_review"
        message = f"{len(urgent_cbc)} urgent CBC/clinical rule flag(s) are present."
    else:
        status = "stable_or_recovering"
        message = "CBC toxicity does not appear worse in later represented cycles."

    return {
        "status": status,
        "message": message,
        "early_min_wbc": round(early_min_wbc, 2),
        "late_min_wbc": round(late_min_wbc, 2),
        "early_min_platelets": round(early_min_platelets, 2),
        "late_min_platelets": round(late_min_platelets, 2),
    }


def _response_trend(mri_signal, report):
    outcome = report.get("treatment_outcome") or {}
    message = mri_signal.get("message") or "No MRI/model response trend is available."
    status = mri_signal.get("status") or "unknown"
    if outcome:
        message = f"{message} Final recorded outcome: {outcome.get('response_category')} / {outcome.get('cancer_status')}."
    return {
        "status": status,
        "message": message,
        "source": mri_signal.get("source"),
        "score": mri_signal.get("response_signal_score"),
    }


def _discordance(mri_signal, clinical_signal):
    favorable_response = mri_signal.get("status") == "favorable_response_signal"
    clinical_review = clinical_signal.get("status") in {"needs_review", "watch_closely"}
    if favorable_response and clinical_review:
        return {
            "status": "response_toxicity_discordance",
            "message": "Response signal is favorable, but CBC/symptom monitoring flags still need review.",
        }
    if not favorable_response and clinical_review:
        return {
            "status": "aligned_concern",
            "message": "Response and clinical monitoring signals both need closer review.",
        }
    return {
        "status": "no_major_discordance",
        "message": "No major response-versus-toxicity discordance is visible in the current summary.",
    }


def _last_n_days(timeline, days):
    dated_events = []
    for event in timeline:
        event_date = _parse_date(event.get("date"))
        if event_date:
            dated_events.append((event_date, event))
    if not dated_events:
        return {"window_days": days, "events": []}

    latest_date = max(item[0] for item in dated_events)
    start_date = latest_date - timedelta(days=days)
    events = [
        event for event_date, event in sorted(dated_events, key=lambda item: item[0], reverse=True)
        if event_date >= start_date
    ]
    return {
        "window_days": days,
        "window_start": start_date.date().isoformat(),
        "window_end": latest_date.date().isoformat(),
        "events": events,
    }


def _missing_data_warnings(report):
    warnings = []
    if not report.get("lab_history"):
        warnings.append("Missing CBC/lab trend data.")
    if not report.get("symptoms"):
        warnings.append("Missing patient-reported symptom data.")
    if not report.get("treatment_effects"):
        warnings.append("Treatment-cycle lab alignment is incomplete.")
    if not report.get("radiology_summary") and not report.get("synthetic_model_prediction"):
        warnings.append("Missing imaging/model response signal.")
    if report.get("has_synthetic_labs"):
        warnings.append("CBC values are synthetic demo data in this project build.")
    return warnings


def _tumor_board_brief(report, assessment):
    summary = report.get("patient_timeline_summary") or {}
    headline = summary.get("headline") or assessment.get("overall_message") or "No headline available."
    key_points = "; ".join((summary.get("key_points") or [])[:4])
    review_flags = "; ".join((summary.get("review_flags") or [])[:3])
    return f"{headline} Key points: {key_points or 'none listed'}. Review flags: {review_flags or 'none listed'}."


def _parse_date(value):
    if not value:
        return None
    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        return datetime(value.year, value.month, value.day)
    try:
        return datetime.fromisoformat(str(value)[:10])
    except ValueError:
        return None
