from backend.services.credibility_gap_registry import build_credibility_gap_registry


def test_registry_separates_internal_and_non_self_certifiable_gaps(tmp_path):
    report = build_credibility_gap_registry(
        output_path=tmp_path / "registry.json",
        doc_path=tmp_path / "registry.md",
    )

    assert report["clinical_validation"] is False
    assert report["healthcare_production_ready"] is False
    assert report["summary"]["gap_count"] >= 10
    assert report["summary"]["cannot_be_self_certified_count"] >= 3
    assert "not a quality" in report["summary"]["score_interpretation"].lower()
    assert (tmp_path / "registry.json").exists()
    assert (tmp_path / "registry.md").exists()


def test_registry_keeps_external_and_clinical_gaps_open(tmp_path):
    report = build_credibility_gap_registry(
        output_path=tmp_path / "registry.json",
        doc_path=tmp_path / "registry.md",
    )
    by_id = {item["id"]: item for item in report["gaps"]}

    for gap_id in (
        "independent_clean_clone_reproduction",
        "external_no_read_evaluation",
        "clinician_and_genetics_review",
        "real_data_irb_clinical_validation",
    ):
        assert by_id[gap_id]["cannot_be_self_certified"] is True
        assert by_id[gap_id]["current_status"] in {
            "blocked_external",
            "blocked_institutional",
        }
    assert by_id["rag_improvement_over_bm25"]["current_status"] != "complete_internal"
    assert "not proven" in by_id["rag_improvement_over_bm25"][
        "honest_claim_until_closed"
    ].lower()


def test_every_gap_has_owner_evidence_completion_and_verification(tmp_path):
    report = build_credibility_gap_registry(
        output_path=tmp_path / "registry.json",
        doc_path=tmp_path / "registry.md",
    )
    for gap in report["gaps"]:
        assert gap["owner"]
        assert gap["evidence_artifacts"]
        assert gap["completion_criteria"]
        assert gap["verification_command"]
        assert gap["honest_claim_until_closed"]


def test_controlled_provider_probe_closes_only_internal_usage_gap(tmp_path):
    report = build_credibility_gap_registry(
        output_path=tmp_path / "registry.json",
        doc_path=tmp_path / "registry.md",
    )
    by_id = {item["id"]: item for item in report["gaps"]}
    provider = by_id["provider_token_usage_coverage"]
    assert provider["current_status"] == "complete_internal"
    assert provider["evidence_snapshot"]["controlled_probe_completed"] is True
    assert provider["evidence_snapshot"][
        "controlled_probe_provider_usage_coverage_rate"
    ] >= 0.8
    assert "not audited billing truth" in provider[
        "honest_claim_until_closed"
    ]
