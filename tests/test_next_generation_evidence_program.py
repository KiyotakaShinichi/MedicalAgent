from __future__ import annotations

import csv
import json
from pathlib import Path

from backend.services.human_review_feedback_ingestion import build_human_review_feedback_ingestion
from backend.services.rag_failure_attribution_vnext import build_rag_failure_attribution
from backend.services.synthetic_load_matrix import run_synthetic_load_matrix
from nlcare_eval.runner import run_evaluation


def test_failure_attribution_maps_section_and_citation_stages(tmp_path: Path) -> None:
    source = tmp_path / "failures.json"
    source.write_text(json.dumps({
        "failures": [
            {"case_id": "a", "failure_reasons": ["expected_section_missing_at_10"]},
            {"case_id": "b", "failure_reasons": ["low_citation_precision"]},
            {"case_id": "a", "failure_reasons": ["expected_section_missing_at_10"]},
        ]
    }), encoding="utf-8")
    result = build_rag_failure_attribution(
        input_paths=[source],
        output_path=tmp_path / "output.json",
    )
    assert result["aggregate_by_stage"]["section_mismatch"]["count"] == 1
    assert result["aggregate_by_stage"]["citation_alignment"]["count"] == 1
    assert result["raw_aggregate_by_stage"]["section_mismatch"] == 2
    assert result["clinical_validation"] is False


def test_human_review_templates_do_not_count_as_completed(tmp_path: Path) -> None:
    template = tmp_path / "reviewer_feedback_template.csv"
    template.write_text("reviewer_role,date\n<role>,YYYY-MM-DD\n", encoding="utf-8")
    result = build_human_review_feedback_ingestion(
        review_dir=tmp_path,
        output_path=tmp_path / "output.json",
    )
    assert result["status"] == "BLOCKED_EXTERNAL"
    assert result["external_review_completed"] is False
    assert result["accepted_feedback_row_count"] == 0


def test_human_review_ingestion_requires_boundary_acknowledgement(tmp_path: Path) -> None:
    path = tmp_path / "external_nurse_feedback.csv"
    fieldnames = [
        "reviewer_role", "date", "artifact_reviewed", "case_or_section_id", "comment",
        "severity", "required_fix", "reviewer_decision", "fix_status",
        "not_clinical_approval_acknowledged",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "reviewer_role": "oncology_nurse", "date": "2026-08-11", "artifact_reviewed": "case.json",
            "case_or_section_id": "case-1", "comment": "Revise wording", "severity": "medium",
            "required_fix": "true", "reviewer_decision": "revise", "fix_status": "pending",
            "not_clinical_approval_acknowledged": "false",
        })
    result = build_human_review_feedback_ingestion(
        review_dir=tmp_path,
        output_path=tmp_path / "output.json",
    )
    assert result["status"] == "needs_attention"
    assert result["accepted_feedback_row_count"] == 0
    assert "clinical_approval_boundary_not_acknowledged" in result["validation_issues"][0]["issues"]


def test_synthetic_load_matrix_runs_declared_levels_without_forbidden_tools(tmp_path: Path) -> None:
    result = run_synthetic_load_matrix(
        output_path=tmp_path / "load.json",
        concurrency_levels=(1, 2),
        requests_per_level=8,
    )
    assert [row["concurrency"] for row in result["profiles"]] == [1, 2]
    assert result["invariants"]["forbidden_tool_exposure_count"] == 0
    assert result["invariants"]["exception_count"] == 0
    assert result["clinical_validation"] is False


def test_nlcare_eval_quick_writes_json_and_markdown(tmp_path: Path, monkeypatch) -> None:
    import nlcare_eval.runner as runner

    monkeypatch.setattr(runner, "_expand_suites", lambda suites: ["fixture"])
    monkeypatch.setattr(runner, "_registry", lambda: {
        "fixture": lambda: {"status": "acceptable", "clinical_validation": False, "pass_rate": 1.0}
    })
    monkeypatch.setattr(runner, "_provenance", lambda: {
        "git_commit": "fixture", "working_tree_dirty": True, "knowledge_base_fingerprint": "fixture"
    })
    output = tmp_path / "run.json"
    markdown = tmp_path / "run.md"
    result = run_evaluation({"quick"}, output_path=output, markdown_path=markdown)
    assert result["status"] == "acceptable_internal_run"
    assert output.exists() and markdown.exists()
    assert "not clinical validation" in result["claim_boundary"].lower()
