from backend.services.patient_xai_envelope import build_patient_xai_envelope


def test_xai_envelope_discloses_missingness_uncertainty_and_noncausality():
    payload = build_patient_xai_envelope(
        prediction={"hybrid_mle_signal": {"hybrid_score": 62.0, "classification_probability": 0.71}},
        explanation={
            "method": "linear_model_coefficient_contribution",
            "positive_contributions": [
                {"feature": "cbc_available", "contribution": 0.2, "meaning": "CBC present"}
            ],
        },
        hybrid_prediction={
            "classification": {
                "decision": "favorable_synthetic_pattern",
                "confidence": "low",
                "model_version": "synthetic-v1",
                "evidence": {
                    "modalities_present": ["cbc"],
                    "modalities_missing": ["imaging"],
                    "sufficiency": "partial",
                    "abstain": False,
                    "confidence_modifier": 0.6,
                },
            }
        },
        data_availability=None,
    )

    assert payload["evidence"]["inputs_used"] == ["cbc"]
    assert payload["evidence"]["inputs_missing"] == ["imaging"]
    assert payload["uncertainty"]["uncertainty_is_clinical_probability"] is False
    assert payload["provenance"]["causal_interpretation_allowed"] is False
    assert payload["top_model_factors"][0]["relative_contribution"] == 0.2
    assert payload["clinical_validation"] is False


def test_xai_envelope_surfaces_abstention():
    payload = build_patient_xai_envelope(
        prediction=None,
        explanation=None,
        hybrid_prediction={"classification": {"evidence": {"abstain": True, "reason": "missing imaging"}}},
        data_availability=None,
    )
    assert payload["status"] == "abstained"
    assert payload["evidence"]["abstain_reason"] == "missing imaging"
