"""Build a resumable human-review packet for fine-tune contamination flags."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_PATH = Path(
    "Data/evals/models/latest_finetune_semantic_contamination.json"
)
DEFAULT_PACKET_PATH = Path(
    "Data/finetune/evaluations/semantic_contamination_adjudication_packet.json"
)
DEFAULT_OUTPUT_PATH = Path(
    "Data/evals/models/latest_finetune_contamination_adjudication_readiness.json"
)
ALLOWED_DECISIONS = {"contaminated", "not_contaminated", "ambiguous"}

CLAIM_BOUNDARY = (
    "This packet is a lexical-semantic reviewer aid. It does not prove absence "
    "of contamination, adapter safety, clinical validity, or patient-facing readiness."
)


def build_finetune_contamination_adjudication(
    source_path: str | Path = DEFAULT_SOURCE_PATH,
    packet_path: str | Path = DEFAULT_PACKET_PATH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _read_json(source_path)
    existing = _read_json(packet_path)
    existing_by_id = {
        str(row.get("pair_id")): row
        for row in existing.get("candidates") or []
        if isinstance(row, dict)
    }
    candidates = []
    for row in source.get("flagged_pairs") or []:
        if not isinstance(row, dict):
            continue
        previous = existing_by_id.get(str(row.get("pair_id"))) or {}
        candidates.append(
            {
                **row,
                "priority": 1 if row.get("severity") == "critical" else 2,
                "decision": previous.get("decision"),
                "reviewer_role": previous.get("reviewer_role"),
                "reviewed_at": previous.get("reviewed_at"),
                "reviewer_notes": previous.get("reviewer_notes"),
            }
        )
    candidates.sort(key=lambda row: (row["priority"], -float(row["max_similarity"])))
    issues = validate_adjudication_candidates(candidates)
    reviewed = sum(row.get("decision") in ALLOWED_DECISIONS for row in candidates)
    unresolved = len(candidates) - reviewed
    completed = bool(candidates) and unresolved == 0 and not issues
    packet = {
        "schema_version": "finetune_contamination_adjudication_packet_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_artifact": str(source_path).replace("\\", "/"),
        "allowed_decisions": sorted(ALLOWED_DECISIONS),
        "review_contract": {
            "reviewer_role_required": True,
            "reviewed_at_required": True,
            "reviewer_notes_required": True,
            "text_is_not_copied_into_packet": True,
            "source_rows_must_be_inspected_locally": True,
        },
        "candidates": candidates,
        "clinical_validation": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    readiness = {
        "schema_version": "finetune_contamination_adjudication_readiness_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed_internal_review" if completed else "ready_for_human_adjudication",
        "completed": completed,
        "source_artifact": str(source_path).replace("\\", "/"),
        "packet_path": str(packet_path).replace("\\", "/"),
        "candidate_count": len(candidates),
        "reviewed_count": reviewed,
        "unresolved_count": unresolved,
        "critical_unresolved_count": sum(
            row.get("severity") == "critical"
            and row.get("decision") not in ALLOWED_DECISIONS
            for row in candidates
        ),
        "counts_by_channel": dict(Counter(row.get("channel") for row in candidates)),
        "validation_issues": issues,
        "adapter_promotion_allowed": False,
        "external_no_read_evaluation_completed": False,
        "clinical_validation": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    return packet, readiness


def write_finetune_contamination_adjudication(
    packet_path: str | Path = DEFAULT_PACKET_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    source_path: str | Path = DEFAULT_SOURCE_PATH,
) -> dict[str, Any]:
    packet, readiness = build_finetune_contamination_adjudication(
        source_path, packet_path
    )
    _write_json(packet_path, packet)
    _write_json(output_path, readiness)
    return readiness


def validate_adjudication_candidates(
    candidates: list[dict[str, Any]],
) -> list[str]:
    issues = []
    seen = set()
    for row in candidates:
        pair_id = str(row.get("pair_id") or "")
        if not pair_id or pair_id in seen:
            issues.append(f"missing_or_duplicate_pair_id:{pair_id}")
        seen.add(pair_id)
        decision = row.get("decision")
        if decision is None:
            continue
        if decision not in ALLOWED_DECISIONS:
            issues.append(f"invalid_decision:{pair_id}")
        for field in ("reviewer_role", "reviewed_at", "reviewer_notes"):
            if not str(row.get(field) or "").strip():
                issues.append(f"missing_{field}:{pair_id}")
    return issues


def _read_json(path: str | Path) -> dict[str, Any]:
    file = Path(path)
    if not file.exists():
        return {}
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = [
    "build_finetune_contamination_adjudication",
    "validate_adjudication_candidates",
    "write_finetune_contamination_adjudication",
]
