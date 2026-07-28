from __future__ import annotations

import json

from backend.services.xai_rank_stability_audit import (
    _group_for_display,
    build_xai_rank_stability_audit,
)


def _fixture(path):
    patients = {}
    for index in range(20):
        patients[f"P{index:02d}"] = {
            "patient_id": f"P{index:02d}",
            "all_contributions": [
                {"feature": "mri_percent_change_from_baseline", "contribution": 5.0},
                {"feature": "pre_wbc", "contribution": 2.0 + index / 100},
                {"feature": "stage_IIA", "contribution": -1.1},
                {"feature": "stage_IIB", "contribution": 0.9},
                {"feature": "symptom_count", "contribution": 0.5},
                {"feature": "dose_delayed", "contribution": 0.2},
            ],
        }
    path.write_text(json.dumps({"patients": patients}), encoding="utf-8")


def test_display_grouping_collapses_one_hot_and_excludes_proxies():
    grouped = _group_for_display(
        {
            "stage_IIA": 1.0,
            "stage_IIB": 2.0,
            "mri_percent_change_from_baseline": 9.0,
            "pre_wbc": 0.5,
        },
        exclude_proxies=True,
    )
    assert grouped["stage"] == 3.0
    assert grouped["pre_wbc"] == 0.5
    assert "mri_percent_change_from_baseline" not in grouped


def test_rank_stability_is_reproducible_and_bounded(tmp_path):
    source = tmp_path / "xai.json"
    _fixture(source)
    first = build_xai_rank_stability_audit(
        source,
        tmp_path / "first.json",
        bootstrap_n=60,
        top_k=4,
        seed=7,
    )
    second = build_xai_rank_stability_audit(
        source,
        tmp_path / "second.json",
        bootstrap_n=60,
        top_k=4,
        seed=7,
    )
    assert first["patient_display_grouped_ranking"] == second[
        "patient_display_grouped_ranking"
    ]
    assert first["clinical_validation"] is False
    assert first["causal_interpretation_allowed"] is False
    assert first["model_retraining_stability_evaluated"] is False
    assert 0 <= first["patient_display_grouped_ranking"]["top_k_jaccard_p05"] <= 1


def test_rank_stability_rejects_tiny_samples(tmp_path):
    source = tmp_path / "tiny.json"
    source.write_text(
        json.dumps(
            {
                "patients": {
                    "P1": {
                        "all_contributions": [
                            {"feature": "pre_wbc", "contribution": 1.0}
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    try:
        build_xai_rank_stability_audit(source, tmp_path / "out.json")
    except ValueError as exc:
        assert "10 patient explanations" in str(exc)
    else:
        raise AssertionError("tiny explanation sample must be rejected")
