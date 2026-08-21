from __future__ import annotations

import json
from pathlib import Path

from backend.services.ml_logic_safety_alignment import build_ml_logic_safety_alignment


def test_ml_logic_safety_alignment_writes_nonclinical_artifact(tmp_path: Path):
    output = tmp_path / "latest_ml_logic_safety_alignment.json"

    build_ml_logic_safety_alignment(output_path=output)

    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "ml_logic_safety_alignment_v1"
    assert payload["clinical_validation"] is False
    assert payload["synthetic_only"] is True
    assert payload["healthcare_production_ready"] is False
    assert "not clinical validation" in payload["claim_boundary"].lower()


def test_ml_logic_safety_alignment_preserves_promotion_boundary():
    report = build_ml_logic_safety_alignment(output_path=None)
    checks = {check["name"]: check for check in report["checks"]}

    promotion = checks["nonclinical_promotion_policy"]
    assert promotion["status"] == "passed"
    assert promotion["evidence"]["may_influence_treatment"] is False
    assert promotion["evidence"]["mle_promotion_gate_decision"] in {"HOLD", "REJECT"}


def test_ml_logic_safety_alignment_surfaces_missing_imaging_gap():
    report = build_ml_logic_safety_alignment(output_path=None)
    checks = {check["name"]: check for check in report["checks"]}

    sufficiency = checks["evidence_sufficiency_alignment"]
    assert sufficiency["status"] in {"passed", "needs_attention", "failed"}
    assert "no_imaging_abstention_rate" in sufficiency["evidence"]
    if sufficiency["evidence"]["no_imaging_drop_detected"]:
        assert sufficiency["status"] == "needs_attention"
        assert "imaging" in sufficiency["recommendation"].lower()


def test_ml_logic_safety_alignment_keeps_toxicity_shortcut_visible():
    report = build_ml_logic_safety_alignment(output_path=None)
    checks = {check["name"]: check for check in report["checks"]}

    shortcut = checks["shortcut_risk_boundaries"]
    assert shortcut["evidence"]["toxicity_policy"] == "review_hint_only"
    assert "toxicity_auroc" in shortcut["evidence"]
    assert "review-hint-only" in shortcut["recommendation"].lower()


def test_ml_logic_safety_alignment_next_steps_from_attention_items():
    report = build_ml_logic_safety_alignment(output_path=None)
    summary = report["summary"]

    assert summary["check_count"] >= 10
    assert summary["logic_alignment_score"] <= 1.0
    assert report["highest_leverage_ml_next_steps"]


def test_ml_logic_safety_alignment_includes_coverage_and_toxicity_v3_checks():
    report = build_ml_logic_safety_alignment(output_path=None)
    checks = {check["name"]: check for check in report["checks"]}

    coverage = checks["coverage_risk_diagnostics"]
    toxicity_v3 = checks["toxicity_target_v3_boundary"]

    assert coverage["status"] == "passed"
    assert coverage["evidence"]["minimum_required_abstention_rate"] >= 0.95
    assert toxicity_v3["status"] == "passed"
    assert toxicity_v3["evidence"]["legacy_rule_does_not_define_v3"] is True
    assert toxicity_v3["evidence"]["promotion_decision"] == "hold_synthetic_only"
