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
