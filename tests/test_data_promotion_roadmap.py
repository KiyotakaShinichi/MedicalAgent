from __future__ import annotations

import json

from backend.services.data_promotion_roadmap import build_data_promotion_roadmap


def test_data_promotion_roadmap_keeps_all_heads_monitor_only(tmp_path):
    output = tmp_path / "roadmap.json"
    report = build_data_promotion_roadmap(output_path=str(output))

    assert output.exists()
    assert report["status"] == "strong"
    assert report["promotion_policy"]["current_global_policy"] == "monitor_only"
    assert report["promotion_policy"]["may_influence_treatment"] is False
    assert "does not establish clinical validity" in report["claim_boundary"]

    by_head = {head["head"]: head for head in report["model_heads"]}
    assert {
        "response_classification",
        "response_regression",
        "toxicity_signal",
        "genetic_biomarker_context",
        "tumor_marker_context",
        "treatment_sequence_context",
    }.issubset(by_head)
    assert by_head["response_classification"]["current_policy"] == "monitor_only"
    assert by_head["toxicity_signal"]["current_policy"] == "review_hint_only"
    assert by_head["tumor_marker_context"]["current_policy"] == "education_and_review_context_only"
    assert by_head["treatment_sequence_context"]["immediate_full_treatment_combo_training_ready"] == 0
    assert by_head["tumor_marker_context"]["tumor_marker_response_train_ready"] == 0
    assert report["cross_head_blockers"]

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["next_big_steps"]) >= 5
    assert payload["next_big_steps"][0]["student_feasible_now"] is True
