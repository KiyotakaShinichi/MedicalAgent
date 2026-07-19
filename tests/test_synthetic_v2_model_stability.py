import pandas as pd

from backend.services.synthetic_v2_model_stability import run_synthetic_v2_model_stability


def _rows():
    rows = []
    for patient in range(24):
        for cycle in (1, 2):
            positive = patient % 2
            rows.append({
                "patient_id": f"P{patient:03d}", "cycle": cycle, "treatment_date": "2026-01-01",
                "age": 40 + patient, "stage": "II" if positive else "I", "molecular_subtype": "HR+/HER2-",
                "regimen": "synthetic regimen", "pre_wbc": 5 + positive, "pre_anc": 2.5,
                "pre_hemoglobin": 12.0, "pre_platelets": 250, "nadir_wbc": 3.0 + positive,
                "nadir_anc": 1.5, "nadir_hemoglobin": 10.5, "nadir_platelets": 180,
                "recovery_wbc": 4.0, "recovery_hemoglobin": 11.0, "recovery_platelets": 220,
                "mri_tumor_size_cm": 3.0 - positive * 0.5, "mri_percent_change_from_baseline": -20,
                "response_score_percent": 25 + positive * 40, "max_symptom_severity": 3 + positive,
                "symptom_count": 2, "intervention_count": 1, "dose_delayed": 0, "dose_reduced": 0,
                "latent_response_strength": 0.7, "toxicity_risk_binary": 0,
                "urgent_intervention_needed": 0, "support_intervention_needed": 0,
                "cycle_response_trend_class": "stable", "final_response_category": "synthetic",
                "final_cancer_status": "synthetic", "treatment_success_binary": positive,
                "maintenance_needed": 0, "final_response_multiclass": "synthetic",
            })
    return rows


def test_stability_eval_is_grouped_and_cannot_promote(tmp_path):
    input_path = tmp_path / "rows.csv"
    pd.DataFrame(_rows()).to_csv(input_path, index=False)
    result = run_synthetic_v2_model_stability(input_path, tmp_path / "result.json")
    assert result["clinical_validation"] is False
    assert result["synthetic_only"] is True
    assert result["promotion_decision"]["decision"] == "HOLD"
    assert result["promotion_decision"]["promotion_allowed"] is False
    assert all(run["patient_overlap_count"] == 0 for run in result["runs"])
    assert len(result["runs"]) == 5 * 6 * 4
    assert len(result["paired_comparisons"]) == 12
    assert all(item["n_paired_seeds"] == 5 for item in result["paired_comparisons"])


def test_target_proxies_are_excluded(tmp_path):
    input_path = tmp_path / "rows.csv"
    pd.DataFrame(_rows()).to_csv(input_path, index=False)
    result = run_synthetic_v2_model_stability(input_path, tmp_path / "result.json")
    assert "response_score_percent" not in result["feature_columns"]
    assert "treatment_success_binary" not in result["feature_columns"]
    assert "mri_percent_change_from_baseline" not in result["feature_columns"]
