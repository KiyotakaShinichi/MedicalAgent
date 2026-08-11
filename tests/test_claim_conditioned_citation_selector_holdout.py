import json
from pathlib import Path

from backend.services.claim_conditioned_citation_selector_holdout import (
    build_selector_holdout_eval,
    freeze_selector_holdout,
)


def _row(case_id: str, *, refusal: bool = False) -> dict:
    return {
        "case_id": case_id,
        "source_case_id": case_id,
        "query": "Explain HER2 generally",
        "answer_text": "HER2 refers to a protein used in breast cancer context.",
        "atomic_claims": ["HER2 refers to a protein used in breast cancer context"],
        "expected_source_ids": ["nci-her2-breast"],
        "refusal_route": refusal,
        "retrieved_chunks": [
            {
                "source_id": "unrelated",
                "text": "CBC includes white blood cells.",
                "source_tier": "T1",
                "allowed_use": "general_patient_education",
                "retrieval_score": 0.9,
            },
            {
                "source_id": "nci-her2-breast",
                "text": "HER2 is a protein used in breast cancer context.",
                "source_tier": "T1",
                "allowed_use": "general_patient_education",
                "retrieval_score": 0.8,
            },
        ],
        "was_used_for_selector_tuning": False,
        "clinical_validation": False,
    }


def test_holdout_eval_is_internal_nonclinical_and_never_promotes_live(tmp_path: Path):
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(
        "".join(json.dumps(_row(f"case-{i}")) + "\n" for i in range(30)),
        encoding="utf-8",
    )
    report = build_selector_holdout_eval(fixture, tmp_path / "report.json")
    assert report["case_count"] == 30
    assert report["was_used_for_selector_tuning"] is False
    assert report["clinical_validation"] is False
    assert report["live_patient_route_changed"] is False
    assert report["promotion_decision"] in {
        "live_shadow_only",
        "offline_only_not_promoted",
    }


def test_refusal_rows_strip_citations(tmp_path: Path):
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(json.dumps(_row("refusal", refusal=True)) + "\n", encoding="utf-8")
    report = build_selector_holdout_eval(fixture, None)
    assert report["refusal_citation_strip_passed"] is True
    assert report["cases"][0]["selected_ids"] == []


def test_freeze_refuses_to_overwrite_existing_fixture(tmp_path: Path):
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(json.dumps(_row("existing")) + "\n", encoding="utf-8")
    before = fixture.read_bytes()
    result = freeze_selector_holdout(fixture, overwrite=False)
    assert result["status"] == "already_frozen"
    assert fixture.read_bytes() == before
