from backend.services.claim_conditioned_citation_selector_eval import (
    build_claim_conditioned_citation_selector_eval,
)


def test_internal_eval_is_improved_but_not_promoted_live(tmp_path):
    report = build_claim_conditioned_citation_selector_eval(tmp_path / "report.json")
    assert report["paired_internal_improvement"] is True
    assert report["selector_citation_precision"] > report["baseline_top3_citation_precision"]
    assert report["selector_claim_support_rate"] >= report["baseline_claim_support_rate"]
    assert report["promotion_decision"] == "offline_shadow_candidate_only"
    assert report["live_patient_route_changed"] is False


def test_eval_discloses_tuning_and_nonclinical_boundaries():
    report = build_claim_conditioned_citation_selector_eval(None)
    assert report["was_used_for_tuning"] is True
    assert report["internal_vs_external"] == "internal_authored_tuning_used"
    assert report["clinical_validation"] is False
    assert report["support_proxy_is_entailment"] is False
    assert report["disallowed_source_selection_count"] == 0
    assert report["refusal_citation_strip_passed"] is True
