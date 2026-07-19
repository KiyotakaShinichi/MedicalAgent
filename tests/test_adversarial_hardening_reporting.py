from backend.services.adversarial_hardening_report import build_adversarial_hardening_report


def test_fresh_v4_is_the_canonical_safety_headline(tmp_path):
    payload = build_adversarial_hardening_report(tmp_path / "report.json")

    assert payload["status"] == "needs_attention"
    assert payload["canonical_headline"]["not_solved"] is True
    assert payload["canonical_headline"]["pass_rate"] == payload["v4_fresh_holdout"]["pass_rate"]
    assert payload["reporting_policy"]["v3_before_after_is_primary"] is False
    assert payload["reporting_policy"]["external_author_eval_completed"] is False
    assert payload["clinical_validation"] is False
