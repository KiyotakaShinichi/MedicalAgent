from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd

from backend.services import mle_readiness


def _passing_check(name: str, category: str) -> dict:
    return mle_readiness._check(
        name=name,
        category=category,
        status="passed",
        value=name,
        threshold="unchanged",
        meaning="behavioral-equivalence marker",
        hard_gate=False,
        remediation="none",
    )


def test_facade_preserves_summary_order_verdicts_and_serialization(tmp_path) -> None:
    check_groups = {
        name: (lambda *args, result=result: result)
        for name, result in {
            "_artifact_checks": [_passing_check("artifact", "artifacts")],
            "_data_contract_checks": [_passing_check("data", "data_contract")],
            "_feature_store_checks": [_passing_check("feature", "feature_store")],
            "_lineage_leakage_holdout_checks": [_passing_check("lineage", "lineage")],
            "_performance_checks": [_passing_check("performance", "model_quality")],
            "_lifecycle_checks": [_passing_check("lifecycle", "lifecycle")],
            "_agent_quality_checks": [_passing_check("agent", "safety_regression")],
            "_robustness_checks": [_passing_check("robustness", "monitoring")],
            "_realism_checks": [_passing_check("realism", "realism")],
        }.items()
    }
    output_path = tmp_path / "mle_readiness.json"
    frame = pd.DataFrame({"patient_id": ["P001"]})

    with (
        patch.object(mle_readiness, "_load_csv", return_value=frame),
        patch.object(mle_readiness, "_load_json", return_value={}),
        patch.object(mle_readiness, "_load_latest_evaluation_report", return_value={}),
        patch.object(mle_readiness, "load_latest_agent_regression_report", return_value={}),
        patch.object(mle_readiness, "_artifact_hashes", return_value=[]),
        patch.object(mle_readiness, "_hybrid_weight_ablation", return_value={"status": "ablation"}),
        patch.object(
            mle_readiness,
            "_temporal_generalization_eval",
            return_value={"status": "temporal"},
        ),
        patch.object(mle_readiness, "run_temporal_eval", return_value={"status": "stable"}),
        patch.object(mle_readiness, "run_noise_eval", return_value={"status": "robust"}),
        patch.object(mle_readiness, "run_calibration_eval", return_value={"status": "passed"}),
        patch.object(
            mle_readiness,
            "build_synthetic_realism_report",
            return_value={"status": "acceptable"},
        ),
        patch.multiple(mle_readiness, **check_groups),
    ):
        summary = mle_readiness.build_mle_readiness_summary(output_path=output_path)

    assert [item["name"] for item in summary["checks"]] == [
        "artifact",
        "data",
        "feature",
        "lineage",
        "performance",
        "lifecycle",
        "agent",
        "robustness",
        "realism",
    ]
    assert summary["status"] == "strong"
    assert summary["release_recommendation"] == "strong_for_engineering_poc_not_clinical_validation"
    assert summary["hard_gate_status"] == "passed"
    assert summary["hard_gate_failures"] == []
    assert summary["poc_demo_readiness"]["status"] == "ready_with_limitations"
    assert summary["next_actions"] == []
    assert summary["hybrid_weight_ablation"] == {"status": "ablation"}
    assert summary["temporal_generalization_eval"] == {"status": "temporal"}

    serialized = json.loads(output_path.read_text(encoding="utf-8"))
    assert list(serialized) == list(summary)
    assert serialized == summary
