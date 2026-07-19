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
    assert result["direction_sign_consistency_rate"] == 1.0
