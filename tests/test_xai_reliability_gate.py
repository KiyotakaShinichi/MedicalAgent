import json

from backend.services.xai_reliability_gate import (
    build_xai_reliability_gate,
    load_xai_reliability_policy,
)


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_gate_suppresses_rank_claim_when_retraining_order_is_unstable(tmp_path):
    fidelity = tmp_path / "fidelity.json"
    rank = tmp_path / "rank.json"
    retraining = tmp_path / "retraining.json"
    _write(
        fidelity,
        {"additivity_pass_rate": 1.0, "finite_contribution_rate": 1.0},
    )
    _write(rank, {"human_participant_study_completed": False})
    _write(
        retraining,
        {
            "metrics": {
                "global_top_k_jaccard_p05": 0.6,
                "global_rank_correlation_median": -0.2,
                "global_rank_correlation_p05": -1.0,
            },
            "consensus_feature_tiers": {
                "stable_core_alphabetical": [
                    {"feature_group": "cycle", "top_k_inclusion_rate": 1.0}
                ],
                "suppressed_low_consensus_alphabetical": [
                    {"feature_group": "age", "top_k_inclusion_rate": 0.2}
                ],
            },
            "presentation_policy": {
                "enforced": True,
                "exact_rank_display_allowed": False,
            },
        },
    )
    result = build_xai_reliability_gate(
        fidelity_path=fidelity,
        rank_path=rank,
        retraining_path=retraining,
        output_path=tmp_path / "out.json",
        doc_path=tmp_path / "out.md",
    )
    policy = result["patient_display_policy"]
    assert policy["mode"] == "grouped_factors_without_rank_claim"
    assert policy["show_grouped_factors"] is True
    assert policy["ranked_feature_order_allowed"] is False
    assert policy["show_numeric_shap_values"] is False
    assert policy["stable_factor_groups"] == ["cycle"]
    assert policy["display_order_basis"] == "alphabetical_not_importance"
    assert result["status"] == "acceptable"
    assert result["status_basis"] == "bounded_consensus_groups_exact_order_suppressed"
    assert result["causal_interpretation_allowed"] is False


def test_missing_artifact_policy_fails_closed(tmp_path):
    policy = load_xai_reliability_policy(tmp_path / "missing.json")
    assert policy["mode"] == "suppress_feature_factors"
    assert policy["show_grouped_factors"] is False
    assert policy["ranked_feature_order_allowed"] is False
    assert policy["stable_factor_groups"] == []
