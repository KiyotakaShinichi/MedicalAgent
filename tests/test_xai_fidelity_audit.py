import json

from backend.services.xai_fidelity_audit import build_xai_fidelity_audit


def test_missing_prediction_and_base_value_are_reported(tmp_path):
    source = {
        "method": "synthetic_shap", "shap_available": True,
        "patients": {"P1": {"patient_id": "P1", "prediction": None,
            "positive_contributions": [{"feature": "stage_II", "contribution": 0.4, "direction": "toward_success"},
                                       {"feature": "stage_III", "contribution": 0.2, "direction": "toward_success"}],
            "negative_contributions": [{"feature": "mri_percent_change_from_baseline", "contribution": -0.3, "direction": "toward_non_success"}]}}
    }
    path = tmp_path / "xai.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    result = build_xai_fidelity_audit(path, tmp_path / "audit.json")
    assert result["status"] == "needs_attention"
    assert result["additivity_verifiable"] is False
    assert result["multiple_one_hot_feature_patient_rate"] == 1.0
    assert result["causal_interpretation_allowed"] is False
    assert result["clinical_validation"] is False


def test_complete_minimal_export_can_verify_mechanics(tmp_path):
    source = {"method": "synthetic_shap", "shap_available": True, "patients": {"P1": {
        "patient_id": "P1", "prediction": 0.6, "base_value": 0.5,
        "positive_contributions": [{"feature": "wbc", "contribution": 0.1, "direction": "toward_success"}],
        "negative_contributions": [],
    }}}
    path = tmp_path / "xai.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    result = build_xai_fidelity_audit(path, tmp_path / "audit.json")
    assert result["additivity_verifiable"] is True
    assert result["additivity_pass_rate"] == 1.0
    assert result["direction_sign_consistency_rate"] == 1.0


def test_full_log_odds_export_verifies_all_contributions(tmp_path):
    source = {"method": "synthetic_shap", "shap_available": True, "patients": {"P1": {
        "patient_id": "P1", "base_value": -0.4,
        "model_output": {"mean_prediction_log_odds": 0.25},
        "all_contributions": [
            {"feature": "wbc", "contribution": 0.8, "direction": "toward_success"},
            {"feature": "nadir_anc", "contribution": -0.15, "direction": "toward_non_success"},
        ],
        "positive_contributions": [], "negative_contributions": [],
    }}}
    path = tmp_path / "xai.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    result = build_xai_fidelity_audit(path, tmp_path / "audit.json")
    assert result["status"] == "acceptable"
    assert result["additivity_verifiable"] is True
    assert result["max_absolute_additivity_residual_log_odds"] == 0.0


def test_additivity_mismatch_is_not_accepted(tmp_path):
    source = {"method": "synthetic_shap", "shap_available": True, "patients": {"P1": {
        "patient_id": "P1", "base_value": 0.0,
        "model_output": {"mean_prediction_log_odds": 1.0},
        "all_contributions": [
            {"feature": "wbc", "contribution": 0.2, "direction": "toward_success"},
        ],
        "positive_contributions": [], "negative_contributions": [],
    }}}
    path = tmp_path / "xai.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    result = build_xai_fidelity_audit(path, tmp_path / "audit.json")
    assert result["status"] == "needs_attention"
    assert result["additivity_verifiable"] is True
    assert result["additivity_pass_rate"] == 0.0
