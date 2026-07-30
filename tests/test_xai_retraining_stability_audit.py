import json

import pandas as pd

from backend.services.complete_synthetic_training import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)
from backend.services.xai_retraining_stability_audit import (
    build_xai_retraining_stability_audit,
)


def _frame() -> pd.DataFrame:
    rows = []
    for patient_index in range(80):
        label = patient_index % 2
        for cycle in range(1, 4):
            row = {
                "patient_id": f"P{patient_index:03d}",
                "treatment_success_binary": label,
                "stage": "IIA" if patient_index % 3 else "IIB",
                "molecular_subtype": "HR+" if label else "TNBC",
                "regimen": "A" if patient_index % 2 else "B",
            }
            for feature_index, feature in enumerate(NUMERIC_FEATURES):
                row[feature] = (
                    label * 0.6
                    + cycle * 0.03
                    + patient_index * 0.001
                    + feature_index * 0.0001
                )
            rows.append(row)
    return pd.DataFrame(rows)


def test_retraining_stability_emits_bounded_nonclinical_artifact(tmp_path) -> None:
    source = tmp_path / "rows.csv"
    output = tmp_path / "audit.json"
    _frame().to_csv(source, index=False)
    payload = build_xai_retraining_stability_audit(
        source,
        output,
        seeds=(3, 5, 7, 11, 13, 17, 19, 23),
        local_patient_limit=20,
    )
    assert payload["model_retraining_stability_evaluated"] is True
    assert payload["local_patient_explanation_stability_evaluated"] is True
    assert payload["clinical_validation"] is False
    assert payload["causal_interpretation_allowed"] is False
    assert payload["seed_count"] == 8
    assert payload["presentation_policy"]["enforced"] is True
    assert "exact_rank_display_allowed" in payload["presentation_policy"]
    assert payload["raw_exact_rank_stability_status"] in {"acceptable", "needs_attention"}
    assert set(payload["consensus_feature_tiers"]) == {
        "stable_core_alphabetical",
        "variable_context_alphabetical",
        "suppressed_low_consensus_alphabetical",
    }
    assert output.exists()
    assert json.loads(output.read_text())["synthetic_only"] is True


def test_retraining_stability_requires_multiple_seeds(tmp_path) -> None:
    source = tmp_path / "rows.csv"
    _frame().to_csv(source, index=False)
    try:
        build_xai_retraining_stability_audit(
            source,
            tmp_path / "audit.json",
            seeds=(1, 2),
        )
    except ValueError as exc:
        assert "eight" in str(exc)
    else:
        raise AssertionError("Expected a seed-count validation error")


def test_fixture_covers_feature_contract() -> None:
    frame = _frame()
    assert not (set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) - set(frame.columns))
