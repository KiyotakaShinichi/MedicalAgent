import csv
import json

from backend.services.synthetic_simple_baseline_audit import build_simple_baseline_audit


def test_simple_baseline_audit_keeps_synthetic_boundary(tmp_path) -> None:
    rows_path = tmp_path / "rows.csv"
    fieldnames = [
        "actual_label", "actual_response_score_percent",
        "logistic_regression_probability", "gradient_boosting_calibrated_probability",
        "ridge_regression_response_score_percent", "random_forest_regressor_response_score_percent",
    ]
    rows = [
        {"actual_label": 0, "actual_response_score_percent": -20, "logistic_regression_probability": 0.2, "gradient_boosting_calibrated_probability": 0.1, "ridge_regression_response_score_percent": -10, "random_forest_regressor_response_score_percent": -18},
        {"actual_label": 1, "actual_response_score_percent": 30, "logistic_regression_probability": 0.8, "gradient_boosting_calibrated_probability": 0.9, "ridge_regression_response_score_percent": 15, "random_forest_regressor_response_score_percent": 28},
    ]
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    paired_path = tmp_path / "paired.json"
    paired_path.write_text(json.dumps({"classification": [{
        "candidate_model": "logistic_regression",
        "method": "exact_mcnemar_binomial_on_paired_correctness",
        "accuracy_delta_champion_minus_candidate": 0.0,
        "p_value": 1.0,
    }]}), encoding="utf-8")

    artifact = build_simple_baseline_audit(rows_path, paired_path)
    assert artifact["total_n"] == 2
    assert artifact["classification"]["constant_half"]["auroc"] == 0.5
    selective = artifact["classification_selective_risk"]["logistic_regression"]
    assert [point["requested_coverage"] for point in selective["points"]] == [1.0, 0.9, 0.75, 0.5]
    assert selective["ranking_signal"] == "absolute_distance_from_probability_0.5"
    regression_curve = artifact["regression_disagreement_abstention"]
    assert regression_curve["calibrated_uncertainty"] is False
    assert len(regression_curve["points"]) == 4
    assert artifact["paired_champion_vs_logistic"]["superiority_proven"] is False
    assert artifact["promotion_allowed"] is False
    assert artifact["clinical_validation"] is False
    assert "real patients" in artifact["claim_boundary"]
    assert "not calibrated clinical uncertainty" in artifact["uncertainty_boundary"]
