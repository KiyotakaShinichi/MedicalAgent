from backend.services.counterfactual_stability import run_counterfactual_stability_eval
from backend.services.medical_claim_boundary import classify_medical_claim
from backend.services.medical_claim_boundary_eval import run_medical_claim_boundary_eval
from backend.services.minimum_evidence import build_minimum_evidence_standards_artifact, evaluate_minimum_evidence
from backend.services.per_head_calibration import run_per_head_calibration
from backend.services.self_supervised_timeline import run_self_supervised_timeline_pretraining
from backend.services.shortcut_audit import run_shortcut_audit
from backend.services.supplement_safety import flag_supplement_safety
from backend.services.synthetic_realism_hardening import build_synthetic_realism_hardening_report
from backend.services.toxicity_review_mapping import map_toxicity_review_hint


def test_synthetic_realism_hardening_reports_frequency_checks(tmp_path):
    payload = build_synthetic_realism_hardening_report(output_path=str(tmp_path / "realism.json"))
    assert payload["summary"]["checks_total"] == 8
    assert "modality_missingness" in payload["checks"]
    assert payload["claim_boundary"].startswith("Synthetic realism")


def test_self_supervised_pretraining_uses_prior_timeline_only(tmp_path):
    payload = run_self_supervised_timeline_pretraining(output_path=str(tmp_path / "ssl.json"))
    assert payload["metrics"]["leakage_check_status"] == "passed"
    assert all(not feature.startswith("future_") for feature in payload["features"])
    assert payload["metrics"]["masked_lab_mae"] >= 0


def test_counterfactual_stability_artifact_has_unacceptable_flip_metric(tmp_path):
    payload = run_counterfactual_stability_eval(output_path=str(tmp_path / "counterfactual.json"), sample_rows=40)
    assert "unacceptable_flip_count" in payload["summary"]
    assert payload["summary"]["scenario_count"] >= 5
    assert payload["gate"]["unacceptable_probability_delta"] > 0


def test_per_head_calibration_reports_all_three_heads(tmp_path):
    payload = run_per_head_calibration(output_path=str(tmp_path / "calibration.json"))
    assert set(payload["heads"]) == {"response_classification", "toxicity", "response_regression", "abstention"}
    assert "brier" in payload["heads"]["response_classification"]
    assert "ece" in payload["heads"]["toxicity"]


def test_shortcut_audit_surfaces_generator_shortcut_risk(tmp_path):
    payload = run_shortcut_audit(output_path=str(tmp_path / "shortcut.json"))
    assert "toxicity_audit" in payload
    assert "regression_audit" in payload
    assert payload["claim_boundary"].startswith("Shortcut audit is synthetic-only")


def test_minimum_evidence_blocks_response_without_evidence(tmp_path):
    payload = build_minimum_evidence_standards_artifact(output_path=str(tmp_path / "standards.json"))
    assert payload["status"] == "strong"
    result = evaluate_minimum_evidence("response_pattern_estimation", ["demographics"])
    assert result["decision"] == "insufficient"
    assert "imaging" in result["missing_required_any_options"]


def test_medical_claim_boundary_blocks_false_reassurance_and_eval_passes(tmp_path):
    blocked = classify_medical_claim("This is safe with chemo and there is no need to worry.")
    assert blocked["decision"] == "blocked"
    assert "false_reassurance" in blocked["blocked_claim_types"]
    payload = run_medical_claim_boundary_eval(output_path=str(tmp_path / "claim_boundary.json"))
    assert payload["status"] == "strong"


def test_toxicity_review_mapping_is_hint_not_grade():
    payload = map_toxicity_review_hint(symptom="fever after chemo", severity=4, anc=0.7)
    assert payload["safe_label"] == "Review severity hint"
    assert payload["review_priority"] == "urgent_review"
    assert "not a clinician-assigned CTCAE grade" in payload["claim_boundary"]


def test_supplement_safety_flags_review_without_calling_safe():
    payload = flag_supplement_safety("St John's wort", current_medications=["tamoxifen"])
    assert payload["status"] == "review_needed"
    assert "st_johns_wort" in payload["matched_categories"]
    assert "does not determine whether a supplement is safe or unsafe" in payload["claim_boundary"]
