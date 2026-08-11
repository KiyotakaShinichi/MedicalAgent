from backend.services.focused_release_summary import build_report


def test_focused_release_summary_is_small_honest_and_non_clinical():
    report = build_report()
    assert report["clinical_validation"] is False
    assert report["healthcare_production_ready"] is False
    assert report["core_evidence"]["rag"]["improvement_proven_vs_bm25"] is False
    assert report["core_evidence"]["rag"]["external_holdout_completed"] is False
    assert report["core_evidence"]["latency"]["production_ready"] is False
    assert len(report["active_warnings"]) >= 5
    assert "retrieval superiority over BM25" in report["what_this_release_cannot_claim"]


def test_focused_release_summary_exposes_confirmed_write_gate():
    report = build_report()
    write = report["core_evidence"]["confirmed_write_contract"]
    assert write["case_count"] >= 50
    assert write["pass_rate"] == 1.0


def test_focused_release_summary_separates_development_from_frozen_evidence():
    report = build_report()
    development = report["core_evidence"]["unsafe_intent_development_controls"]
    architecture = report["core_evidence"]["architecture_maintainability"]

    assert development["classifier_pass_rate"] == 1.0
    assert development["mutation_pass_rate"] == 1.0
    assert development["safe_negative_pass_rate"] == 1.0
    assert development["used_for_tuning"] is True
    assert development["independent_evidence"] is False
    assert architecture["oversized_file_count"] <= 9
    assert architecture["critical_file_count"] == 0
