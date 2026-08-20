from __future__ import annotations

import pandas as pd

from backend.services.mle_readiness import (
    _hybrid_weight_ablation,
    _temporal_generalization_eval,
)
from backend.services.mle_readiness_statistics import (
    hybrid_weight_ablation,
    temporal_generalization_eval,
)


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": [f"P{index:02d}" for index in range(12)],
            "actual_label": [0, 1] * 6,
            "model_calibrated_probability": [
                0.1,
                0.9,
                0.2,
                0.8,
                0.3,
                0.7,
                0.25,
                0.75,
                0.15,
                0.85,
                0.35,
                0.65,
            ],
            "model_response_score_percent": [-30, 30] * 6,
        }
    )


def test_hybrid_ablation_is_reexported_without_contract_change() -> None:
    predictions = _prediction_frame()
    direct = hybrid_weight_ablation(predictions)
    reexported = _hybrid_weight_ablation(predictions)
    assert reexported == direct
    assert direct["status"] == "available"
    assert len(direct["sweep"]) == 13
    assert direct["warning"] == "Synthetic data only - not clinical evidence."


def test_temporal_generalization_is_reexported_without_contract_change() -> None:
    predictions = _prediction_frame()
    training = pd.DataFrame(
        {
            "patient_id": predictions["patient_id"],
            "cycle": [1] * 6 + [2] * 6,
        }
    )
    direct = temporal_generalization_eval(training, predictions)
    reexported = _temporal_generalization_eval(training, predictions)
    assert reexported == direct
    assert direct["status"] == "stable"
    assert "Synthetic data only" in direct["warning"]


def test_statistics_fail_closed_to_unavailable_when_evidence_is_missing() -> None:
    empty = pd.DataFrame()
    assert hybrid_weight_ablation(empty)["status"] == "unavailable"
    assert temporal_generalization_eval(empty, empty)["status"] == "unavailable"
