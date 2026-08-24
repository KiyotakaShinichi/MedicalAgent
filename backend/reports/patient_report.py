"""Assembly of the patient report payload returned by the API.

This is the boundary where pandas-derived data becomes a JSON response, so it
is also where pandas' missing-value sentinels have to stop.

A NULL text column does not survive the trip as None. Under the string dtype a
missing value is represented as float `nan`, so a symptom with no notes arrives
here as a float sitting in a text field, and `json.dumps` rejects it because
NaN is not JSON. The endpoint answered 500 for any patient with a blank note.

The DataFrames themselves are deliberately left alone. Missing numeric labs are
meaningful to `analyze_labs` and `detect_risks`, which rely on NaN propagating
through their arithmetic; scrubbing it upstream would change clinical output to
fix a serialization problem. So the sentinels are converted once, here, on the
way out.
"""

import math

import pandas as pd

# pd.NA and NaT cannot be compared with `!=` or truth-tested safely, so they are
# matched by identity rather than with a predicate.
_MISSING_SENTINELS = (pd.NA, pd.NaT)


def _json_safe(value):
    """Return `value` with anything JSON cannot represent replaced by None.

    Deliberately narrow: strings, bools, ints and finite floats come back
    unchanged and identical, and only non-finite floats and pandas' missing
    sentinels are rewritten. Containers are walked so a sentinel nested in a
    risk's evidence or a timeline entry cannot slip past, but nothing is
    coerced, reformatted, or reordered along the way.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        # Covers numpy floats too, which subclass float. Infinity is as
        # unserializable as NaN and becomes null for the same reason.
        return value if math.isfinite(value) else None
    if any(value is sentinel for sentinel in _MISSING_SENTINELS):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_json_safe(item) for item in value)
    return value


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
    lab_sources = sorted({row.get("source", "unknown") for row in lab_history})
    has_synthetic_labs = any(source.startswith("synthetic") for source in lab_sources)

    return _json_safe({
        "patient_state": patient_state,
        "latest_labs": latest_labs,
        "baseline_labs": baseline_labs,
        "lab_history": lab_history,
        "lab_sources": lab_sources,
        "has_synthetic_labs": has_synthetic_labs,
        "trends": trends,
        "risks": risks,
        "treatment_effects": treatment_effects,
        "radiology_summary": radiology_summary,
        "breast_imaging_summary": radiology_summary,
        "symptoms": symptoms.to_dict(orient="records") if symptoms is not None and not symptoms.empty else [],
        "timeline": timeline,
        "ai_summary": ai_summary,
        "safety_note": "Breast cancer clinical decision-support only. Not for diagnosis, cancer detection, confirming metastasis, or replacing a licensed clinician.",
    })
