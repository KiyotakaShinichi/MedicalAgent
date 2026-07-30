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
    assert len(report["decision_threshold_sensitivity"]) == 5
    assert len(report["exported_probability_stability"]["rows"]) == 3
    assert len(report["prevalence_reweighting_sensitivity"]) == 3
    assert len(report["subgroup_slices"]) >= 3
    assert all("accuracy_wilson_95" in row for row in report["subgroup_slices"])
    assert report["paired_baseline_comparison"]["test"] == "exact_two_sided_mcnemar"
    assert report["paired_baseline_comparison"]["paired_accuracy_delta"]["replicates"] >= 1000
    assert (
        report["headline"]["champion_superiority_over_logistic_proven"] is False
    )
    assert report["status"] == "needs_attention"


def test_statistical_audit_is_synthetic_only_and_non_promotional():
    report = build_report()
    assert report["clinical_validation"] is False
    assert report["synthetic_only"] is True
    assert report["healthcare_production_ready"] is False
    assert report["promotion_decision"] == "hold_synthetic_only"
    assert "do not recompute model predictions" in " ".join(report["limitations"])
    assert "not clinical evidence" in report["claim_boundary"]
