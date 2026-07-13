"""External red-team readiness artifact.

Counts only real, committed external-author adversarial case files.
Refuses to fabricate completed reviews under any circumstance.

Output: ``Data/evals/governance/latest_external_red_team_readiness.json``
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REVIEW_DIR = Path("Data/evals/external_review")
OUTPUT_PATH = Path("Data/evals/governance/latest_external_red_team_readiness.json")
TEMPLATE_PATH = REVIEW_DIR / "adversarial_case_submission_template.jsonl"
QUICKSTART_PATH = Path("docs/review_packets/external_adversarial_red_team_quickstart.md")


_INTERNAL_AUTHOR_TOKENS = frozenset({
    "engineering",
    "oncotrack_team",
    "oncotrack_team+claude_codex",
    "engineering_internal",
})


def _load_case_file(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _validate_row(row: dict[str, Any]) -> str | None:
    """Return a disqualification reason or None when the row counts."""
    if str(row.get("case_source")) != "external_author_red_team_v1":
        return "case_source mismatch"
    if bool(row.get("was_used_for_tuning", True)):
        return "was_used_for_tuning is true"
    author = str(row.get("authored_by") or "").strip().lower()
    if not author or author in _INTERNAL_AUTHOR_TOKENS:
        return f"authored_by is internal or missing: {author!r}"
    text = json.dumps(row, ensure_ascii=False)
    if "<PLACEHOLDER" in text or "PLACEHOLDER:" in text:
        return "placeholder marker present"
    return None


def build_readiness() -> dict[str, Any]:
    case_files: list[dict[str, Any]] = []
    attestation_files: list[str] = []
    if REVIEW_DIR.exists():
        for path in REVIEW_DIR.glob("adversarial_cases_*_*.jsonl"):
            if path.name == TEMPLATE_PATH.name:
                continue
            rows = _load_case_file(path)
            qualified: list[dict[str, Any]] = []
            disqualified: list[dict[str, Any]] = []
            for row in rows:
                reason = _validate_row(row)
                if reason:
                    disqualified.append({"case_id": row.get("case_id"), "reason": reason})
                else:
                    qualified.append(row)
            case_files.append({
                "path": str(path).replace("\\", "/"),
                "n_rows": len(rows),
                "n_qualified": len(qualified),
                "n_disqualified": len(disqualified),
                "disqualified_rows": disqualified[:20],
            })
        # Attestations: any non-template attestation file qualifies.
        for path in REVIEW_DIR.glob("*_attestation.md"):
            if path.name == "reviewer_attestation_template.md":
                continue
            text = path.read_text(encoding="utf-8").lower()
            if "[x]" in text:
                attestation_files.append(str(path).replace("\\", "/"))

    completed = sum(f["n_qualified"] for f in case_files)
    has_template = TEMPLATE_PATH.exists()
    has_quickstart = QUICKSTART_PATH.exists()
    status = (
        "ready_to_request_review"
        if has_template and has_quickstart and completed == 0
        else "in_progress" if completed > 0 else "needs_attention"
    )

    return {
        "schema_version": "external_red_team_readiness_v1",
        "status": status,
        "label": "external_red_team_readiness",
        "clinical_validation": False,
        "claim_boundary": (
            "External red-team readiness.  Counts ONLY real external-author "
            "adversarial cases committed to "
            "Data/evals/external_review/adversarial_cases_<role>_<date>.jsonl. "
            "Does NOT fabricate completed reviews.  Not clinical validation; "
            "not clinician sign-off."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "template_present": has_template,
        "quickstart_present": has_quickstart,
        "template_path": str(TEMPLATE_PATH).replace("\\", "/"),
        "quickstart_path": str(QUICKSTART_PATH).replace("\\", "/"),
        "completed_external_cases": completed,
        "completed_reviews": completed,
        "n_attestation_files_found": len(attestation_files),
        "attestation_files": attestation_files,
        "case_files": case_files,
        "anti_fabrication_invariant": (
            "completed_external_cases is derived from row-level checks on real "
            "JSONL files.  An empty or template-shaped file MUST NOT increment "
            "this count.  Filling in the submission template with placeholder "
            "rows is treated as a release-gate failure."
        ),
    }


def write_readiness(output_path: Path = OUTPUT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(build_readiness(), indent=2), encoding="utf-8")
    return output_path


__all__ = ["OUTPUT_PATH", "build_readiness", "write_readiness"]
