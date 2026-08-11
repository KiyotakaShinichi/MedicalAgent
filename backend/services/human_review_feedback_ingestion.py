"""Validate structured external-review feedback without fabricating review."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = ROOT / "Data/evals/external_review"
DEFAULT_OUTPUT = ROOT / "Data/evals/governance/latest_human_review_feedback_ingestion.json"
TEMPLATE_NAME = "reviewer_feedback_template.csv"
REQUIRED_COLUMNS = {
    "reviewer_role",
    "date",
    "artifact_reviewed",
    "case_or_section_id",
    "comment",
    "severity",
    "reviewer_decision",
    "fix_status",
    "not_clinical_approval_acknowledged",
}
ALLOWED_SEVERITIES = {"low", "medium", "high", "blocker"}
ALLOWED_DECISIONS = {"keep", "revise", "move", "split", "ambiguous", "not_applicable"}
ALLOWED_FIX_STATUS = {"pending", "in_progress", "wont_fix_with_rationale", "fixed"}
CLAIM_BOUNDARY = (
    "Independent review of selected scenarios does not establish clinical validation, clinical effectiveness, "
    "regulatory approval, or production healthcare readiness. No review is counted from a template or incomplete row."
)


def build_human_review_feedback_ingestion(
    *,
    review_dir: str | Path = REVIEW_DIR,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    directory = Path(review_dir)
    candidate_files = sorted(
        path for path in directory.glob("*.csv")
        if path.name != TEMPLATE_NAME and "template" not in path.name.lower()
    ) if directory.exists() else []
    accepted_rows: list[dict[str, str]] = []
    rejected_rows: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for path in candidate_files:
        accepted_before = len(accepted_rows)
        rejected_before = len(rejected_rows)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing_columns = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or []))
            if missing_columns:
                rejected_rows.append({"file": _relative(path), "row": None, "issues": [f"missing_columns:{','.join(missing_columns)}"]})
            else:
                for row_number, raw in enumerate(reader, start=2):
                    row = {str(key): str(value or "").strip() for key, value in raw.items()}
                    issues = _validate_row(row)
                    if issues:
                        rejected_rows.append({"file": _relative(path), "row": row_number, "issues": issues})
                    else:
                        accepted_rows.append(row)
        files.append({
            "path": _relative(path),
            "accepted_rows": len(accepted_rows) - accepted_before,
            "rejected_rows": len(rejected_rows) - rejected_before,
        })

    completed = bool(accepted_rows)
    status = "review_feedback_ingested" if completed and not rejected_rows else (
        "needs_attention" if rejected_rows else "BLOCKED_EXTERNAL"
    )
    payload = {
        "schema_version": "human_review_feedback_ingestion_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "clinical_validation": False,
        "external_review_completed": completed,
        "candidate_file_count": len(candidate_files),
        "accepted_feedback_row_count": len(accepted_rows),
        "rejected_feedback_row_count": len(rejected_rows),
        "files": files,
        "accepted_feedback": accepted_rows,
        "validation_issues": rejected_rows,
        "issue_summary": _issue_summary(accepted_rows),
        "next_step": (
            "Triage accepted findings into dated issues and add regression tests for accepted fixes."
            if completed else
            "Send the prepared packets to eligible independent reviewers and store returned CSV files in Data/evals/external_review/."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _validate_row(row: dict[str, str]) -> list[str]:
    issues: list[str] = []
    for field in REQUIRED_COLUMNS:
        if not row.get(field) or row[field].startswith("<"):
            issues.append(f"missing_or_placeholder:{field}")
    if row.get("severity") not in ALLOWED_SEVERITIES:
        issues.append("invalid_severity")
    if row.get("reviewer_decision") not in ALLOWED_DECISIONS:
        issues.append("invalid_reviewer_decision")
    if row.get("fix_status") not in ALLOWED_FIX_STATUS:
        issues.append("invalid_fix_status")
    if row.get("not_clinical_approval_acknowledged", "").lower() != "true":
        issues.append("clinical_approval_boundary_not_acknowledged")
    return issues


def _issue_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    severities = {severity: 0 for severity in sorted(ALLOWED_SEVERITIES)}
    required_fix = 0
    for row in rows:
        severities[row["severity"]] += 1
        required_fix += row.get("required_fix", "").lower() == "true"
    return {"by_severity": severities, "required_fix_count": required_fix}


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


__all__ = ["build_human_review_feedback_ingestion"]
