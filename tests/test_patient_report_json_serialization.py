"""Pandas missing-value sentinels must not reach the patient report payload.

`GET /me/patient-report/core` answered 500 for any patient with a blank note.
The report is assembled from DataFrames, and under the string dtype a missing
text value is a float `nan`, so a NULL note arrived in the payload as a float
and strict JSON serialization refused it.

The endpoint tests below go through the real route, so they exercise the
serializer FastAPI actually uses rather than asserting against a dict that
never gets encoded. The unit tests around them pin the conversion itself,
including the cases this patient's data does not happen to contain.
"""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.routers import patient as patient_router
from backend.reports.patient_report import _json_safe, build_patient_report


client = TestClient(app)

ENRICHMENT_STUB = {"status": "queued", "retry_after_ms": 750}


def _patient_token() -> str:
    response = client.post(
        "/auth/demo-credential-login",
        json={"username": "P001", "password": "patient-demo"},
    )
    assert response.status_code == 200
    return response.json()["access_token"]


def _core_report(monkeypatch):
    """Fetch the core report through the real route."""
    monkeypatch.setattr(
        patient_router,
        "_schedule_report_enrichment",
        lambda patient_id: dict(ENRICHMENT_STUB),
    )
    token = _patient_token()
    return client.get(
        "/me/patient-report/core",
        headers={"Authorization": f"Bearer {token}"},
    )


def _report(*, symptoms: pd.DataFrame | None = None, labs: pd.DataFrame | None = None):
    """Assemble a report from just the frames a test cares about."""
    empty = pd.DataFrame()
    return build_patient_report(
        patient_state={},
        labs=labs if labs is not None else empty,
        trends={},
        risks=[],
        treatment_effects=[],
        radiology_summary=None,
        symptoms=symptoms if symptoms is not None else empty,
        timeline=[],
        ai_summary="",
    )


def _assert_reproduces_the_sentinel(frame: pd.DataFrame, column: str, row: int) -> None:
    """Guard against a fixture that no longer reproduces the bug.

    Only a *mixed* text column triggers this: pandas gives it the `str` dtype,
    whose missing value is a float `nan`. A column of nothing but None stays
    `object` and keeps None, which serializes fine and would let these tests
    pass without exercising the fix at all.
    """
    value = frame[column].iloc[row]
    assert isinstance(value, float) and math.isnan(value), (
        f"fixture no longer produces a NaN sentinel in {column!r} "
        f"(dtype={frame[column].dtype}, value={value!r})"
    )


def _symptoms(*notes: object) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": f"2026-01-{index + 1:02d}",
                "symptom": "fatigue",
                "severity": 3,
                "notes": note,
            }
            for index, note in enumerate(notes)
        ]
    )


def _labs(*source_notes: object) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": f"2026-01-{index + 1:02d}",
                "wbc": 6.0,
                "hemoglobin": 12.0,
                "platelets": 250.0,
                "source": "synthetic",
                "source_note": note,
            }
            for index, note in enumerate(source_notes)
        ]
    )


# --- A, B, C: missing text becomes null -------------------------------------


def test_missing_notes_becomes_null() -> None:
    symptoms = _symptoms("recorded", None)
    _assert_reproduces_the_sentinel(symptoms, "notes", 1)

    report = _report(symptoms=symptoms)
    assert report["symptoms"][1]["notes"] is None


def test_missing_source_note_becomes_null() -> None:
    labs = _labs("recorded", None)
    _assert_reproduces_the_sentinel(labs, "source_note", 1)

    report = _report(labs=labs)
    assert report["lab_history"][1]["source_note"] is None


def test_both_missing_become_null_and_serialize() -> None:
    symptoms = _symptoms("recorded", None)
    labs = _labs("recorded", None)
    _assert_reproduces_the_sentinel(symptoms, "notes", 1)
    _assert_reproduces_the_sentinel(labs, "source_note", 1)

    report = _report(symptoms=symptoms, labs=labs)

    assert report["symptoms"][1]["notes"] is None
    assert report["lab_history"][1]["source_note"] is None
    json.dumps(report, allow_nan=False, default=str)


def test_a_missing_note_is_null_not_the_string_nan() -> None:
    """The tempting wrong fix: "nan" is valid JSON and silently wrong."""
    symptoms = _symptoms("recorded", None)
    _assert_reproduces_the_sentinel(symptoms, "notes", 1)

    notes = _report(symptoms=symptoms)["symptoms"][1]["notes"]
    assert notes is None
    assert notes != "nan"
    assert notes != "NaN"


# --- D: real values survive untouched ---------------------------------------


def test_present_text_is_returned_unchanged() -> None:
    original = "Mild nausea after cycle 1; synthetic demo data."
    report = _report(symptoms=_symptoms(original, None))
    assert report["symptoms"][0]["notes"] == original
    assert report["symptoms"][1]["notes"] is None


def test_finite_numbers_are_preserved_exactly() -> None:
    report = _report(labs=_labs("note"))
    row = report["lab_history"][0]
    assert row["wbc"] == 6.0
    assert row["hemoglobin"] == 12.0
    assert row["platelets"] == 250.0


def test_zero_and_false_are_not_treated_as_missing() -> None:
    """A falsy value is not an absent one; a truthiness check would drop these."""
    assert _json_safe(0) == 0
    assert _json_safe(0.0) == 0.0
    assert _json_safe(False) is False
    assert _json_safe("") == ""


# --- E, F, G: the real endpoint ---------------------------------------------


def test_core_report_endpoint_succeeds(monkeypatch) -> None:
    response = _core_report(monkeypatch)
    assert response.status_code == 200, response.text[:400]


def test_core_report_survives_strict_json(monkeypatch) -> None:
    """The regression guard: this raises ValueError on the pre-fix payload."""
    response = _core_report(monkeypatch)
    assert response.status_code == 200

    json.dumps(response.json(), allow_nan=False, default=str)


def test_core_report_body_contains_no_nan_literal(monkeypatch) -> None:
    """Standards-compliant JSON carries no NaN or Infinity tokens."""
    response = _core_report(monkeypatch)
    assert response.status_code == 200

    body = response.content.decode("utf-8")
    for literal in ("NaN", "Infinity", "-Infinity"):
        assert literal not in body, f"{literal} escaped into the response body"


def test_an_absent_note_reaches_the_client_as_null(monkeypatch) -> None:
    """End to end: the field is present and null, not missing and not "nan"."""
    response = _core_report(monkeypatch)
    assert response.status_code == 200

    symptoms = response.json()["symptoms"]
    assert symptoms, "P001 should have symptom records"
    assert any(entry["notes"] is None for entry in symptoms), (
        "this fixture is meant to include at least one blank note"
    )
    for entry in symptoms:
        assert "notes" in entry
        assert entry["notes"] is None or isinstance(entry["notes"], str)


# --- the conversion itself --------------------------------------------------


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_numbers_become_null(value: float) -> None:
    assert _json_safe(value) is None


def test_pandas_missing_sentinels_become_null() -> None:
    assert _json_safe(pd.NA) is None
    assert _json_safe(pd.NaT) is None


def test_sentinels_are_reached_inside_nested_structures() -> None:
    """Risks and timeline entries carry notes nested several levels down."""
    payload = {
        "risks": [{"evidence": {"notes": float("nan"), "date": "2026-01-01"}}],
        "timeline": [{"detail": {"notes": float("nan"), "severity": 3}}],
    }
    cleaned = _json_safe(payload)

    assert cleaned["risks"][0]["evidence"]["notes"] is None
    assert cleaned["risks"][0]["evidence"]["date"] == "2026-01-01"
    assert cleaned["timeline"][0]["detail"]["notes"] is None
    assert cleaned["timeline"][0]["detail"]["severity"] == 3


def test_container_types_are_preserved() -> None:
    assert isinstance(_json_safe([1, 2]), list)
    assert isinstance(_json_safe((1, 2)), tuple)
    assert isinstance(_json_safe({"a": 1}), dict)


def test_unknown_objects_pass_through_untouched() -> None:
    """The helper is a serializer guard, not a domain rewriter."""
    marker = object()
    assert _json_safe(marker) is marker


def test_report_assembly_leaves_the_source_frame_alone() -> None:
    """Normalising the response must not mutate the analytics inputs.

    `analyze_labs` and `detect_risks` run on these frames and rely on NaN
    propagating through their arithmetic, which is why the conversion happens
    on the way out rather than at the source.
    """
    symptoms = _symptoms("recorded", None)
    _assert_reproduces_the_sentinel(symptoms, "notes", 1)

    _report(symptoms=symptoms)

    _assert_reproduces_the_sentinel(symptoms, "notes", 1)
