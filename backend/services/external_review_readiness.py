"""External-review readiness artifact for unreviewed evaluation packets.

This module does not claim that external review has happened.  It only checks
that the repo contains enough templates, instructions, and tracking fields to
start independent case authoring and lightweight expert review.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_PATH = Path("Data/evals/governance/latest_external_review_readiness.json")

REQUIRED_FILES = {
    "external_author_rag_instructions": Path("docs/evals/how_to_author_external_rag_cases.md"),
    "external_author_adversarial_instructions": Path("docs/evals/how_to_author_external_adversarial_cases.md"),
    "external_author_eval_packet": Path("docs/review_packets/external_author_eval_packet.md"),
    "rag_case_template": Path("Data/evals/review_templates/external_author_rag_cases_template.jsonl"),
    "adversarial_case_template": Path("Data/evals/review_templates/external_author_adversarial_cases_template.jsonl"),
    "clinical_safety_review_log": Path("Data/evals/review_templates/clinical_safety_review_log_template.csv"),
    "genetics_review_log": Path("Data/evals/review_templates/genetics_review_log_template.csv"),
    "mle_review_log": Path("Data/evals/review_templates/mle_review_log_template.csv"),
}

REQUIRED_TEMPLATE_FIELDS = {
    "case_id",
    "expected_route",
    "expected_refusal_or_escalation",
    "authored_by",
    "authored_date",
    "reviewer_role",
    "was_used_for_tuning",
    "contamination_note",
}

FIELD_ALIASES = {
    "query": {"query", "user_query"},
    "expected_route": {"expected_route", "expected_intent"},
}

CLAIM_BOUNDARY = (
    "External review readiness means packets/templates are prepared. It does "
    "not mean external review, clinician review, clinical validation, or "
    "approval has occurred."
)


def build_external_review_readiness(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    files = [_file_row(name, path) for name, path in REQUIRED_FILES.items()]
    template_rows = [_template_row(name, path) for name, path in REQUIRED_FILES.items() if path.suffix == ".jsonl"]
    missing_files = [row for row in files if not row["exists"]]
    missing_template_fields = [row for row in template_rows if row["missing_required_fields"]]
    completed_review_logs = _completed_review_logs()

    status = "ready_for_external_authoring" if not missing_files and not missing_template_fields else "needs_attention"
    payload = {
        "schema_version": "external_review_readiness_v1_2026_05",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "headline_metric": f"{len(files) - len(missing_files)}/{len(files)} required packet files present",
        "total_n": len(files),
        "pass_count": len(files) - len(missing_files),
        "fail_count": len(missing_files),
        "skipped_count": 0,
        "files": files,
        "template_field_checks": template_rows,
        "completed_external_review_count": len(completed_review_logs),
        "external_author_eval_completed": False,
        "clinician_review_completed": False,
        "genetic_counselor_review_completed": False,
        "senior_mle_review_completed": False,
        "recommended_reviewer_roles": [
            "oncology nurse or clinician reviewer for safety wording",
            "genetic counselor or genetics-trained reviewer for VUS/genetics cases",
            "senior MLE or AI engineer for eval design and leakage skepticism",
            "patient advocate or nontechnical reviewer for overtrust/usability language",
        ],
        "next_steps": [
            "Recruit reviewers and ask them to author cases before reading prompts or internals.",
            "Run authored cases once as a baseline before any tuning.",
            "Record reviewer role, date, comments, severity, linked artifact, and fix status.",
            "Report external-authored results separately from internal regression scores.",
        ],
        "clinical_validation": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "contamination_note": (
            "This artifact is a readiness checklist only. External-author templates "
            "are intentionally blank/examples until independent reviewers author cases."
        ),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _file_row(name: str, path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    return {
        "name": name,
        "path": str(path).replace("\\", "/"),
        "exists": path.exists(),
        "nonempty": bool(text.strip()),
        "mentions_unreviewed": "unreviewed" in text.lower() or "not clinical validation" in text.lower(),
        "mentions_no_clinical_validation": "clinical validation" in text.lower(),
    }


def _template_row(name: str, path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "name": name,
            "path": str(path).replace("\\", "/"),
            "exists": False,
            "observed_fields": [],
            "missing_required_fields": sorted(REQUIRED_TEMPLATE_FIELDS | {"query"}),
        }
    observed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                observed.update(json.loads(line).keys())
            except json.JSONDecodeError:
                continue
            break
    return {
        "name": name,
        "path": str(path).replace("\\", "/"),
        "exists": True,
        "observed_fields": sorted(observed),
        "missing_required_fields": _missing_fields(observed),
    }


def _missing_fields(observed: set[str]) -> list[str]:
    missing = set(REQUIRED_TEMPLATE_FIELDS - observed)
    for canonical, aliases in FIELD_ALIASES.items():
        if observed & aliases:
            missing.discard(canonical)
        else:
            missing.add(canonical)
    return sorted(missing)


def _completed_review_logs() -> list[dict[str, Any]]:
    # Future real review logs should be copied into Data/evals/review_logs/.
    folder = Path("Data/evals/review_logs")
    if not folder.exists():
        return []
    return [
        {"path": str(path).replace("\\", "/"), "name": path.name}
        for path in sorted(folder.glob("*"))
        if path.is_file() and path.stat().st_size > 0
    ]


__all__ = ["build_external_review_readiness"]
