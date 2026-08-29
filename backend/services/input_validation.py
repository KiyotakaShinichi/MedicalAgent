"""Domain validation for patient-entered records.

These bounds are enforced at two layers, on purpose:

* :mod:`backend.api.schemas.patient` declares them as Pydantic field
  constraints, so an HTTP request that violates one is rejected at the boundary
  with a 422 before any handler code runs;
* the ``validate_*`` functions below enforce the same bounds for callers that
  never pass through a request body - notably the support agent's record-write
  actions, which construct records from a conversation.

That is why both layers exist. To stop them drifting into two different answers,
the bounds themselves are declared once here, as the constants below, and the
schema imports them rather than repeating the numbers.

The functions also return *warnings* - values inside the accepted range that a
clinician should still look at. Pydantic cannot express that, and it is not a
rejection, so it stays here.
"""

from __future__ import annotations

from typing import TypedDict

# ─── bounds, declared once and shared with the Pydantic schemas ──────────────

#: Free-text symptom label.
SYMPTOM_MAX_LENGTH = 80
#: Patient-reported severity, inclusive.
SEVERITY_MIN = 0
SEVERITY_MAX = 10
#: Free-text notes attached to a symptom record.
NOTES_MAX_LENGTH = 800
#: A single chat message.
CHAT_MESSAGE_MAX_LENGTH = 3000


class CbcLimits(TypedDict):
    min: float
    max: float
    watch_low: float
    watch_high: float
    unit: str


CBC_LIMITS: dict[str, CbcLimits] = {
    "wbc": {
        "min": 0.1,
        "max": 200.0,
        "watch_low": 2.0,
        "watch_high": 50.0,
        "unit": "10^9/L",
    },
    "hemoglobin": {
        "min": 2.0,
        "max": 25.0,
        "watch_low": 7.0,
        "watch_high": 18.5,
        "unit": "g/dL",
    },
    "platelets": {
        "min": 1.0,
        "max": 2000.0,
        "watch_low": 50.0,
        "watch_high": 1000.0,
        "unit": "10^9/L",
    },
}


def validate_cbc_values(wbc, hemoglobin, platelets):
    values = {
        "wbc": wbc,
        "hemoglobin": hemoglobin,
        "platelets": platelets,
    }
    warnings = []
    for name, value in values.items():
        _require_number(name, value)
        limits = CBC_LIMITS[name]
        numeric = float(value)
        if numeric < limits["min"] or numeric > limits["max"]:
            raise ValueError(
                f"{name}={numeric:g} is outside accepted demo constraints "
                f"({limits['min']}-{limits['max']} {limits['unit']})."
            )
        if numeric <= limits["watch_low"]:
            warnings.append({
                "field": name,
                "level": "clinician_review",
                "message": f"{name} is very low and should be reviewed in clinical context.",
            })
        elif numeric >= limits["watch_high"]:
            warnings.append({
                "field": name,
                "level": "clinician_review",
                "message": f"{name} is very high and should be reviewed in clinical context.",
            })
    return warnings


def validate_symptom_payload(symptom, severity, notes=None):
    _require_text("symptom", symptom, max_length=SYMPTOM_MAX_LENGTH)
    if not isinstance(severity, int):
        raise ValueError(
            f"severity must be an integer between {SEVERITY_MIN} and {SEVERITY_MAX}."
        )
    if severity < SEVERITY_MIN or severity > SEVERITY_MAX:
        raise ValueError(f"severity must be between {SEVERITY_MIN} and {SEVERITY_MAX}.")
    _optional_text("notes", notes, max_length=NOTES_MAX_LENGTH)
    warnings = []
    if severity >= 8:
        warnings.append({
            "field": "severity",
            "level": "clinician_review",
            "message": "High-severity symptoms should be reviewed by the care team.",
        })
    return warnings


def validate_treatment_payload(cycle, drug):
    if not isinstance(cycle, int):
        raise ValueError("cycle must be an integer.")
    if cycle < 1 or cycle > 30:
        raise ValueError("cycle must be between 1 and 30.")
    _require_text("drug", drug, max_length=160)
    return []


def validate_imaging_report_payload(modality, report_type, findings, impression, body_site=None):
    _require_text("modality", modality, max_length=80)
    _require_text("report_type", report_type, max_length=80)
    _optional_text("body_site", body_site, max_length=80)
    _require_text("findings", findings, max_length=12000)
    _require_text("impression", impression, max_length=4000)
    warnings = []
    if len(findings.strip()) < 20 or len(impression.strip()) < 10:
        warnings.append({
            "field": "report_text",
            "level": "insufficient_detail",
            "message": "Imaging report text is short; trend NLP may be limited.",
        })
    return warnings


def validate_patient_payload(patient_id, name):
    _require_text("patient_id", patient_id, max_length=80)
    _require_text("name", name, max_length=160)


def validate_chat_message(message):
    _require_text("message", message, max_length=CHAT_MESSAGE_MAX_LENGTH)


def validation_error_payload(error, route=None):
    return {
        "error_state": "invalid_data",
        "message": str(error),
        "route": route,
        "next_step": "Correct the input value and submit again. Clinician review is still required for medical interpretation.",
    }


def _require_number(name, value):
    if value is None:
        raise ValueError(f"{name} is required.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number.") from exc
    if numeric != numeric:
        raise ValueError(f"{name} cannot be NaN.")


def _require_text(name, value, max_length):
    if value is None or not str(value).strip():
        raise ValueError(f"{name} is required.")
    if len(str(value)) > max_length:
        raise ValueError(f"{name} must be {max_length} characters or less.")


def _optional_text(name, value, max_length):
    if value is not None and len(str(value)) > max_length:
        raise ValueError(f"{name} must be {max_length} characters or less.")
