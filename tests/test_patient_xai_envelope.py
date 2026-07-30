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
        reliability_policy={
            "mode": "ranked_factors_with_noncausal_boundary",
            "show_grouped_factors": True,
            "ranked_feature_order_allowed": True,
            "show_numeric_shap_values": True,
            "maximum_factor_count": 3,
            "warning": "Internal synthetic stability only.",
        },
    )

    assert payload["evidence"]["inputs_used"] == ["cbc"]
    assert payload["evidence"]["inputs_missing"] == ["imaging"]
    assert payload["uncertainty"]["uncertainty_is_clinical_probability"] is False
    assert payload["provenance"]["causal_interpretation_allowed"] is False
    assert payload["top_model_factors"][0]["relative_contribution"] == 0.2
    assert payload["explanation_reliability"]["ranked_feature_order_allowed"] is True
    assert payload["clinical_validation"] is False


def test_xai_envelope_surfaces_abstention():
    payload = build_patient_xai_envelope(
        prediction=None,
        explanation=None,
        hybrid_prediction={"classification": {"evidence": {"abstain": True, "reason": "missing imaging"}}},
        data_availability=None,
        reliability_policy={
            "mode": "suppress_feature_factors",
            "show_grouped_factors": False,
            "ranked_feature_order_allowed": False,
            "show_numeric_shap_values": False,
            "maximum_factor_count": 0,
            "warning": "Unavailable.",
        },
    )
    assert payload["status"] == "abstained"
    assert payload["evidence"]["abstain_reason"] == "missing imaging"


def test_xai_envelope_removes_numeric_rank_claim_for_unstable_policy():
    payload = build_patient_xai_envelope(
        prediction={"hybrid_mle_signal": {"hybrid_score": 55}},
        explanation={
            "positive_contributions": [
                {"feature": "cbc", "contribution": 0.8, "meaning": "CBC context"}
            ]
        },
        hybrid_prediction={"classification": {"evidence": {"abstain": False}}},
        data_availability=None,
        reliability_policy={
            "mode": "grouped_factors_without_rank_claim",
            "show_grouped_factors": True,
            "ranked_feature_order_allowed": False,
            "show_numeric_shap_values": False,
            "maximum_factor_count": 3,
            "warning": "Order unstable.",
        },
    )
    assert payload["top_model_factors"][0]["relative_contribution"] is None
    assert payload["top_model_factors"][0]["rank_interpretation_allowed"] is False
