from __future__ import annotations

import json
from pathlib import Path

from backend.services.ml_coverage_risk_diagnostics import (
    REQUIRED_ABSTENTION_SCENARIOS,
    build_ml_coverage_risk_diagnostics,
)


def test_ml_coverage_risk_diagnostics_writes_nonclinical_artifact(tmp_path: Path):
    output = tmp_path / "coverage_risk.json"

    report = build_ml_coverage_risk_diagnostics(output_path=output)

    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "ml_coverage_risk_diagnostics_v1"
    assert payload["clinical_validation"] is False
    assert payload["synthetic_only"] is True
    assert payload["healthcare_production_ready"] is False
    assert payload["promotion_decision"] == "hold_synthetic_only"
    assert "not clinical validation" in payload["claim_boundary"].lower()


def test_required_low_evidence_scenarios_abstain():
    report = build_ml_coverage_risk_diagnostics(output_path=None)
    required = report["required_abstention_scenarios"]

    assert required["all_required_scenarios_passed"] is True
    assert required["minimum_required_abstention_rate"] >= 0.95
    names = {row["scenario"] for row in required["scenarios"]}
    assert names == set(REQUIRED_ABSTENTION_SCENARIOS)


def test_selective_risk_curve_is_available_and_bounded():
    report = build_ml_coverage_risk_diagnostics(output_path=None)
    selective = report["selective_risk"]

    assert selective["point_count"] >= 5
    assert selective["nonincreasing_coverage"] is True
    assert selective["covered_accuracy_lift_highest_margin"] is not None
    assert "not clinical performance" in selective["interpretation"].lower()


def test_imaging_only_context_does_not_claim_patient_outcomes():
    report = build_ml_coverage_risk_diagnostics(output_path=None)
    text = report["imaging_only_context"]["interpretation"].lower()

    assert "synthetic" in text
    assert "does not authorize" in text
    assert "treatment" in text
