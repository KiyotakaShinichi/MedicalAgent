from backend.services.route_aware_rag_policy_eval import (
    BM25,
    FULL_GOVERNED,
    build_report,
    configuration_for_intent,
)


def test_policy_uses_simple_retrieval_only_for_low_risk_routes():
    assert configuration_for_intent("education") == BM25
    assert configuration_for_intent("portal_help") == BM25
    for intent in ("urgent_escalation", "privacy_refusal", "genetic_counselor_review", "tumor_marker_boundary"):
        assert configuration_for_intent(intent) == FULL_GOVERNED


def test_route_aware_report_is_explicitly_post_hoc_and_non_clinical():
    report = build_report()
    assert report["clinical_validation"] is False
    assert report["healthcare_production_ready"] is False
    assert report["external_author_eval_completed"] is False
    assert report["was_used_for_tuning"] is True
    assert report["policy_pre_registered_before_goldset_inspection"] is False
    assert report["live_patient_route_changed"] is False
    assert report["promotion_decision"] == "hold_for_external_holdout"


def test_route_aware_report_covers_same_internal_cases_and_core_metrics():
    report = build_report()
    assert report["route_aware_summary"]["case_count"] == report["input_goldset_total_n"]
    assert sum(report["policy"]["route_counts"].values()) == report["input_goldset_total_n"]
    for key in (
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "ndcg_at_10",
        "citation_precision",
        "claim_support_rate",
        "unsupported_context_rate",
        "refusal_correct",
        "source_tier_correct",
        "latency_p50_ms",
        "latency_p95_ms",
    ):
        assert key in report["route_aware_summary"]
