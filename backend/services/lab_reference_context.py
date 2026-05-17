from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CLAIM_BOUNDARY = (
    "Reference context is for monitoring support only. It is not a diagnosis, "
    "treatment recommendation, or replacement for lab-specific and clinician-reviewed ranges."
)


@dataclass(frozen=True)
class ReferenceRange:
    low: float
    high: float
    unit: str
    source: str


def build_cbc_reference_context(
    *,
    wbc: float,
    hemoglobin: float,
    platelets: float,
    sex: str | None = None,
    age_years: int | None = None,
    oncology_monitoring: bool = True,
) -> dict[str, Any]:
    """Return patient-safe CBC range context.

    The current patient schema does not store sex/age, so most calls use
    population defaults. When future demographics are available, this function
    already supports the most important hemoglobin lower-bound adjustment.
    """

    ranges = {
        "wbc": ReferenceRange(4.0, 11.0, "K/uL", "population_default"),
        "hemoglobin": _hemoglobin_range(sex),
        "platelets": ReferenceRange(150.0, 400.0, "K/uL", "population_default"),
    }
    values = {
        "wbc": float(wbc),
        "hemoglobin": float(hemoglobin),
        "platelets": float(platelets),
    }
    lab_context = {
        name: {
            "value": value,
            "unit": ranges[name].unit,
            "reference_range": {"low": ranges[name].low, "high": ranges[name].high},
            "status": _classify_lab(name, value, ranges[name]),
            "range_source": ranges[name].source,
        }
        for name, value in values.items()
    }
    return {
        "schema_version": "cbc_reference_context_v1",
        "demographics_used": {
            "sex": _normalise_sex(sex) or "not_available",
            "age_years": age_years if age_years is not None else "not_available",
        },
        "context_type": "oncology_monitoring" if oncology_monitoring else "general_reference",
        "labs": lab_context,
        "limitations": [
            "Population defaults may differ from the reference ranges printed by a specific laboratory.",
            "Hemoglobin ranges vary by sex, age, pregnancy status, altitude, and clinical context.",
            "Chemotherapy monitoring thresholds can be more conservative than general population reference ranges.",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _hemoglobin_range(sex: str | None) -> ReferenceRange:
    normalised = _normalise_sex(sex)
    if normalised == "male":
        return ReferenceRange(13.5, 17.5, "g/dL", "sex_adjusted_population_default")
    if normalised == "female":
        return ReferenceRange(12.0, 15.5, "g/dL", "sex_adjusted_population_default")
    return ReferenceRange(12.0, 17.5, "g/dL", "broad_population_default_no_demographics")


def _normalise_sex(sex: str | None) -> str | None:
    if not sex:
        return None
    lowered = sex.strip().lower()
    if lowered in {"m", "male", "man"}:
        return "male"
    if lowered in {"f", "female", "woman"}:
        return "female"
    return None


def _classify_lab(name: str, value: float, ref: ReferenceRange) -> str:
    critical_low = {"wbc": 1.0, "hemoglobin": 7.0, "platelets": 20.0}
    critical_high = {"wbc": 50.0, "hemoglobin": 20.0, "platelets": 1000.0}
    borderline_margin = {"wbc": 0.5, "hemoglobin": 0.5, "platelets": 25.0}

    if value <= critical_low[name]:
        return "critical_low_review"
    if value >= critical_high[name]:
        return "critical_high_review"
    if value < ref.low:
        return "low"
    if value > ref.high:
        return "high"
    margin = borderline_margin[name]
    if value <= ref.low + margin or value >= ref.high - margin:
        return "borderline"
    return "within_population_range"


__all__ = ["build_cbc_reference_context", "CLAIM_BOUNDARY"]
