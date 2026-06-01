import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOLDSET_PATH = ROOT / "Data/evals/rag/retrieval_goldset.jsonl"

REQUIRED_FIELDS = {
    "case_id",
    "query",
    "user_query",
    "expected_intent",
    "category",
    "category_tags",
    "gold_source_ids",
    "expected_source_ids",
    "acceptable_source_tiers",
    "contradiction_traps",
    "expected_allowed_use",
    "expected_answerability_status",
    "expected_refusal_or_insufficient_evidence",
    "authored_by",
    "authored_date",
    "internal_vs_external_authored",
    "was_used_for_tuning",
    "contamination_note",
    "clinical_validation",
}

REQUIRED_TAGS = {
    "easy_education",
    "hard_contradiction",
    "no_evidence",
    "taglish",
    "genetics_vus",
    "tumor_marker",
    "supplement",
    "urgent_symptom",
    "source_tier_filtering",
}


def _load_goldset() -> list[dict]:
    return [
        json.loads(line)
        for line in GOLDSET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_retrieval_goldset_size_schema_and_freeze_metadata():
    rows = _load_goldset()

    assert 50 <= len(rows) <= 100
    assert len({row["case_id"] for row in rows}) == len(rows)

    for row in rows:
        assert REQUIRED_FIELDS <= set(row)
        assert row["clinical_validation"] is False
        assert row["was_used_for_tuning"] is False
        assert row["internal_vs_external_authored"] in {"internal", "external", "mixed"}
        assert isinstance(row["category_tags"], list)
        assert row["category"] in row["category_tags"]
        assert isinstance(row["expected_source_ids"], list)
        assert row["expected_source_ids"]
        assert isinstance(row["acceptable_source_tiers"], list)
        assert row["acceptable_source_tiers"]


def test_retrieval_goldset_required_category_coverage():
    rows = _load_goldset()
    tag_counts = Counter(tag for row in rows for tag in row.get("category_tags", []))

    missing = REQUIRED_TAGS - set(tag_counts)
    assert not missing
    for tag in REQUIRED_TAGS:
        assert tag_counts[tag] >= 5


def test_no_evidence_cases_route_to_boundary_or_insufficient_evidence():
    rows = _load_goldset()
    no_evidence_rows = [row for row in rows if "no_evidence" in row.get("category_tags", [])]

    assert len(no_evidence_rows) >= 5
    for row in no_evidence_rows:
        assert row["expected_answerability_status"] in {"insufficient_evidence", "refuse_due_to_safety"}
        assert row["expected_refusal_or_insufficient_evidence"] is True
        assert "Project safety policy" in row["expected_source_ids"]
