from __future__ import annotations

from pathlib import Path

from backend.services.toxicity_review_target_v3 import TARGET, run_toxicity_review_target_v3


def test_toxicity_review_target_v3_is_synthetic_review_only(tmp_path: Path):
    report = run_toxicity_review_target_v3(output_path=str(tmp_path / "toxicity_v3.json"))

    assert report["schema_version"] == "toxicity_review_target_v3_v1"
    assert report["clinical_validation"] is False
    assert report["synthetic_only"] is True
    assert report["healthcare_production_ready"] is False
    assert report["target"] == TARGET
    assert report["recommendation"]["production_policy"] == "review_hint_only"
    assert report["recommendation"]["promotion_decision"] == "hold_synthetic_only"
    assert "not clinical validation" in report["claim_boundary"].lower()


def test_toxicity_review_target_v3_reduces_legacy_rule_dominance(tmp_path: Path):
    report = run_toxicity_review_target_v3(output_path=str(tmp_path / "toxicity_v3.json"))
    shortcut = report["shortcut_comparison"]

    assert shortcut["legacy_rule_does_not_define_v3"] is True
    assert shortcut["legacy_rule_accuracy_against_v3"] < 0.85
    assert shortcut["legacy_rule_auroc_against_v3"] < 0.90


def test_toxicity_review_target_v3_keeps_shortcut_warning_visible(tmp_path: Path):
    report = run_toxicity_review_target_v3(output_path=str(tmp_path / "toxicity_v3.json"))
    correlations = report["feature_group_sensitivity"]["correlations_with_v3_score"]

    assert correlations
    assert report["shortcut_comparison"]["residual_shortcut_warning"] in {True, False}
    assert "shortcut-risk warnings" in report["feature_group_sensitivity"]["interpretation"]


def test_toxicity_review_target_v3_blocks_clinical_claims(tmp_path: Path):
    report = run_toxicity_review_target_v3(output_path=str(tmp_path / "toxicity_v3.json"))
    unsupported = {item.lower() for item in report["recommendation"]["not_supported"]}

    assert "clinical toxicity prediction" in unsupported
    assert "ctcae grade assignment" in unsupported
    assert "patient-facing treatment action" in unsupported
    assert "real adverse-event detection" in unsupported
