"""Patient-readable comparison of confirmed portal records.

This module compares recorded values only. It deliberately does not infer
treatment effectiveness, disease status, prognosis, or causality.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Mapping, Sequence

from backend.services.lab_reference_context import build_cbc_reference_context


SCHEMA_VERSION = "record_change_explanation_v1"


def build_record_change_explanation(
    *,
    lab_history: Sequence[Mapping[str, Any]] | None = None,
    symptoms: Sequence[Mapping[str, Any]] | None = None,
    imaging_reports: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    missing: list[str] = []
    directional_votes: list[int] = []

    lab_observation = _compare_labs(lab_history or [])
    if lab_observation:
        observations.append(lab_observation)
        directional_votes.append(int(lab_observation["review_direction_vote"]))
    else:
        missing.append("At least two dated CBC records are needed for a CBC comparison.")

    symptom_observation = _compare_symptoms(symptoms or [])
    if symptom_observation:
        observations.append(symptom_observation)
        directional_votes.append(int(symptom_observation["review_direction_vote"]))
    else:
        missing.append("Symptoms need entries on at least two dates for a severity comparison.")

    imaging_observation = _compare_imaging(imaging_reports or [])
    if imaging_observation:
        observations.append(imaging_observation)
        directional_votes.append(int(imaging_observation["review_direction_vote"]))
    else:
        missing.append("At least two dated imaging measurements are needed for a size comparison.")

    status = _status(directional_votes)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "headline": _headline(status),
        "patient_summary": _patient_summary(status),
        "observations": [
            {key: value for key, value in item.items() if key != "review_direction_vote"}
            for item in observations
        ],
        "missing_or_not_comparable": missing,
        "safe_next_steps": _safe_next_steps(status),
        "comparison_basis": (
            "Latest confirmed portal entry versus the previous comparable confirmed entry. "
            "CBC band counts use fixed demonstration bands; imaging compares recorded measurements; "
            "symptoms compare the highest recorded severity per date."
        ),
        "treatment_effectiveness_conclusion_allowed": False,
        "clinical_validation": False,
        "claim_boundary": (
            "Record-change summary only. It does not show whether treatment is working, establish "
            "cancer response or progression, recommend treatment, or replace clinician review."
        ),
    }


def _compare_labs(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    by_date: dict[date, Mapping[str, Any]] = {}
    for row in rows:
        observed = _date_value(row.get("date"))
        if observed is not None and _complete_cbc(row):
            by_date[observed] = row
    dates = sorted(by_date)
    if len(dates) < 2:
        return None
    previous, latest = by_date[dates[-2]], by_date[dates[-1]]
    previous_count = _cbc_demo_band_count(previous)
    latest_count = _cbc_demo_band_count(latest)
    vote = _sign(latest_count - previous_count)
    values = []
    for key, label in (("wbc", "WBC"), ("hemoglobin", "hemoglobin"), ("platelets", "platelets")):
        before = _number(previous.get(key))
        after = _number(latest.get(key))
        if before is not None and after is not None:
            values.append(f"{label} {before:g} to {after:g}")
    return {
        "evidence_type": "cbc",
        "previous_date": _iso_date(previous.get("date")),
        "latest_date": _iso_date(latest.get("date")),
        "summary": (
            f"The latest CBC changed from the prior record ({'; '.join(values) or 'values not comparable'}). "
            f"Fixed demo-band matches changed from {previous_count} to {latest_count}."
        ),
        "review_direction": _direction_label(vote),
        "review_direction_vote": vote,
        "calculation": "Counts WBC, hemoglobin, and platelet values outside fixed portal demonstration bands.",
    }


def _compare_symptoms(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    by_date: dict[date, list[float]] = {}
    for row in rows:
        observed = _date_value(row.get("date"))
        severity = _number(row.get("severity"))
        if observed is not None and severity is not None:
            by_date.setdefault(observed, []).append(severity)
    dates = sorted(by_date)
    if len(dates) < 2:
        return None
    previous_date, latest_date = dates[-2], dates[-1]
    previous_peak = max(by_date[previous_date])
    latest_peak = max(by_date[latest_date])
    vote = _sign(latest_peak - previous_peak)
    return {
        "evidence_type": "symptoms",
        "previous_date": previous_date.isoformat(),
        "latest_date": latest_date.isoformat(),
        "summary": (
            f"The highest recorded symptom severity changed from {previous_peak:g}/10 "
            f"to {latest_peak:g}/10 across the two latest symptom dates."
        ),
        "review_direction": _direction_label(vote),
        "review_direction_vote": vote,
        "calculation": "Compares the maximum patient-entered severity on each of the two latest symptom dates.",
    }


def _compare_imaging(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    by_date: dict[date, tuple[float, Mapping[str, Any]]] = {}
    for row in rows:
        observed = _date_value(row.get("date"))
        size = _number(row.get("largest_tumor_size_cm"))
        if observed is not None and size is not None:
            by_date[observed] = (size, row)
    dates = sorted(by_date)
    if len(dates) < 2:
        return None
    previous_date, latest_date = dates[-2], dates[-1]
    previous_size, _ = by_date[previous_date]
    latest_size, _ = by_date[latest_date]
    vote = _sign(latest_size - previous_size)
    return {
        "evidence_type": "imaging_report_measurement",
        "previous_date": previous_date.isoformat(),
        "latest_date": latest_date.isoformat(),
        "summary": (
            f"The largest measurement written in the two latest comparable imaging reports "
            f"changed from {previous_size:g} cm to {latest_size:g} cm."
        ),
        "review_direction": _direction_label(vote),
        "review_direction_vote": vote,
        "calculation": "Compares measurements extracted from report text; it does not inspect raw images.",
    }


def _cbc_demo_band_count(row: Mapping[str, Any]) -> int:
    context = build_cbc_reference_context(
        wbc=float(row["wbc"]),
        hemoglobin=float(row["hemoglobin"]),
        platelets=float(row["platelets"]),
    )
    return sum(
        1
        for lab in context["labs"].values()
        if lab["status"] != "within_population_range"
    )


def _complete_cbc(row: Mapping[str, Any]) -> bool:
    return all(_number(row.get(key)) is not None for key in ("wbc", "hemoglobin", "platelets"))


def _status(votes: Sequence[int]) -> str:
    nonzero = [vote for vote in votes if vote]
    if not votes:
        return "insufficient_comparison_history"
    if not nonzero:
        return "no_clear_record_change"
    if any(vote < 0 for vote in nonzero) and any(vote > 0 for vote in nonzero):
        return "mixed_or_uncertain_record_change"
    return "fewer_logged_review_concerns" if sum(nonzero) < 0 else "more_logged_review_concerns"


def _headline(status: str) -> str:
    return {
        "insufficient_comparison_history": "Not enough comparable history yet",
        "no_clear_record_change": "No clear change across comparable records",
        "mixed_or_uncertain_record_change": "Recorded changes point in different directions",
        "fewer_logged_review_concerns": "Fewer portal review concerns in comparable entries",
        "more_logged_review_concerns": "More portal review concerns in comparable entries",
    }[status]


def _patient_summary(status: str) -> str:
    lead = {
        "insufficient_comparison_history": "The portal cannot compare enough dated entries yet.",
        "no_clear_record_change": "The comparable entries do not show a clear change in portal review signals.",
        "mixed_or_uncertain_record_change": "Some recorded changes have fewer review concerns while others have more.",
        "fewer_logged_review_concerns": "The latest comparable entries have fewer fixed portal review concerns.",
        "more_logged_review_concerns": "The latest comparable entries have more fixed portal review concerns.",
    }[status]
    return f"{lead} This does not show whether treatment is working; the care team must interpret the full clinical record."


def _safe_next_steps(status: str) -> list[str]:
    first = (
        "Review the newer record items with the care team, especially any new or higher-severity concern."
        if status in {"more_logged_review_concerns", "mixed_or_uncertain_record_change"}
        else "Use the comparison to prepare questions for the care team; do not change treatment from this summary."
    )
    return [
        first,
        "Check that dates, units, symptom severity, and copied imaging wording are accurate.",
        "Add missing records only from an existing report or result; do not estimate values.",
    ]


def _direction_label(vote: int) -> str:
    if vote < 0:
        return "fewer_fixed_review_concerns"
    if vote > 0:
        return "more_fixed_review_concerns"
    return "no_change_in_fixed_review_concerns"


def _sign(value: float) -> int:
    return -1 if value < 0 else 1 if value > 0 else 0


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _date_value(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except (TypeError, ValueError):
        return None


def _iso_date(value: Any) -> str | None:
    parsed = _date_value(value)
    return parsed.isoformat() if parsed else None


__all__ = ["SCHEMA_VERSION", "build_record_change_explanation"]
