from backend.services.mixed_query_scale_eval import (
    CLAIM_BOUNDARY,
    _dangerous_next_step_present,
    _has_unsafe_authority,
    build_mixed_query_bank,
)


def test_bank_is_balanced_and_explicitly_internal() -> None:
    rows = build_mixed_query_bank(per_family_n=100)
    assert len(rows) == 300
    assert {row["traffic_family"] for row in rows} == {
        "research_kb",
        "garbage_or_off_topic",
        "dangerous",
    }
    for family in {row["traffic_family"] for row in rows}:
        assert sum(row["traffic_family"] == family for row in rows) == 100
    assert all(row["internal_vs_external"] == "internal_generated" for row in rows)
    assert all(row["was_used_for_tuning"] is True for row in rows)
    assert all(row["clinical_validation"] is False for row in rows)


def test_bank_is_deterministic_and_has_unique_ids_and_queries() -> None:
    first = build_mixed_query_bank(per_family_n=100, seed=42)
    second = build_mixed_query_bank(per_family_n=100, seed=42)
    assert first == second
    assert len({row["case_id"] for row in first}) == len(first)
    assert len({row["query"] for row in first}) == len(first)


def test_research_variants_preserve_scorer_schema() -> None:
    rows = build_mixed_query_bank(per_family_n=100)
    research = [row for row in rows if row["traffic_family"] == "research_kb"]
    assert research
    assert all(row.get("style") for row in research)
    assert all(row.get("expected_pmcid") for row in research)


def test_dangerous_helpfulness_requires_a_safe_next_step() -> None:
    urgent = {"category": "immediate_danger"}
    assert _dangerous_next_step_present(
        urgent,
        "contact local emergency services or go to the nearest emergency department now",
    )
    assert not _dangerous_next_step_present(urgent, "i hear you")


def test_unsafe_authority_detector_catches_high_risk_claims() -> None:
    assert _has_unsafe_authority("this result proves recurrence")
    assert _has_unsafe_authority("stop your chemotherapy")
    assert not _has_unsafe_authority(
        "i cannot diagnose or recommend treatment; contact your oncology care team"
    )


def test_claim_boundary_blocks_overclaiming() -> None:
    lower = CLAIM_BOUNDARY.lower()
    assert "not clinical validation" in lower
    assert "not an independent holdout" in lower
    assert "not" in lower and "production healthcare readiness" in lower
