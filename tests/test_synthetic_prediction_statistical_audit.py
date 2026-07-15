from backend.services.synthetic_prediction_statistical_audit import (
    BOOTSTRAP_REPLICATES,
    PERTURBATION_SEEDS,
    build_report,
)


def test_statistical_audit_uses_row_level_predictions_and_reports_uncertainty():
    report = build_report()
    assert report["total_n"] >= 100
    assert report["patient_count"] == report["total_n"]
    assert report["patient_level_bootstrap"]["replicates"] == BOOTSTRAP_REPLICATES
    assert report["controlled_outcome_perturbations"]["seed_count"] == PERTURBATION_SEEDS
    assert len(report["selective_risk_curve"]) >= 5
    assert len(report["subgroup_slices"]) >= 3
    assert report["paired_baseline_comparison"]["test"] == "exact_two_sided_mcnemar"


def test_statistical_audit_is_synthetic_only_and_non_promotional():
    report = build_report()
    assert report["clinical_validation"] is False
    assert report["synthetic_only"] is True
    assert report["healthcare_production_ready"] is False
    assert report["promotion_decision"] == "hold_synthetic_only"
    assert "do not recompute model predictions" in " ".join(report["limitations"])
    assert "not clinical evidence" in report["claim_boundary"]
