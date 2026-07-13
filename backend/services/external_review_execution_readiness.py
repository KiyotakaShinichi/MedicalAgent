"""External review execution readiness artifact.

Reports whether the reviewer outreach + intake + execution
infrastructure is in place AND counts how many real reviewer
engagements have been filed.  Does NOT fabricate completed reviews
under any circumstances.

Output: ``Data/evals/governance/latest_external_review_execution_readiness.json``
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REVIEW_DIR = Path("Data/evals/external_review")
OUTPUT_PATH = Path("Data/evals/governance/latest_external_review_execution_readiness.json")


# Templates that MUST exist for readiness to report ``ready_to_request_review``.
REQUIRED_TEMPLATE_PATHS: tuple[Path, ...] = (
    Path("docs/review_packets/reviewer_outreach_message_templates.md"),
    Path("docs/review_packets/review_execution_checklist.md"),
    Path("Data/evals/external_review/reviewer_intake_template.md"),
    Path("Data/evals/external_review/reviewer_attestation_template.md"),
    Path("Data/evals/external_review/reviewer_feedback_template.csv"),
)


# Review packets that must exist per reviewer role.  Mirrors
# ``docs/review_packets/INDEX.md``.
REVIEWER_ROLE_PACKETS: dict[str, Path] = {
    "external_peer_engineer":          Path("docs/review_packets/external_author_eval_packet.md"),
    "senior_mle_or_ai_engineer":       Path("docs/review_packets/senior_mle_eval_review_packet.md"),
    "oncology_nurse_or_resident":      Path("docs/review_packets/nurse_or_clinician_safety_review_packet.md"),
    "genetic_counselor":               Path("docs/review_packets/genetic_counselor_vus_review_packet.md"),
    "patient_advocate_or_usability":   Path("docs/review_packets/agentic_workflow_review_packet.md"),
}


PENDING_REVIEW_TYPES: tuple[str, ...] = (
    "held_out_rag_authoring",
    "source_filter_drop_adjudication",
    "adversarial_case_authoring",
    "clinician_safety_review",
    "genetic_counselor_vus_review",
    "senior_mle_review",
)


@dataclass(frozen=True)
class PreparedPacket:
    role: str
    packet_path: Path
    exists: bool


def _count_committed_attestations() -> int:
    """Real attestation files in ``Data/evals/external_review/``.

    The template file is excluded — its presence is not a completed
    review.  Any filename matching ``*_attestation.md`` (except the
    template itself) is treated as a candidate.  The candidate must
    contain the verbatim phrase ``boundary acknowledgements`` AND a
    populated ``reviewer_role`` line.
    """
    if not REVIEW_DIR.exists():
        return 0
    template = REVIEW_DIR / "reviewer_attestation_template.md"
    count = 0
    for path in REVIEW_DIR.glob("*_attestation.md"):
        if path.resolve() == template.resolve():
            continue
        text = path.read_text(encoding="utf-8").lower()
        if "boundary acknowledgements" in text and "reviewer_role" in text:
            # Filled attestations must have at least one ticked checkbox.
            if "[x]" in text:
                count += 1
    return count


def _prepared_packets() -> list[PreparedPacket]:
    return [
        PreparedPacket(role=role, packet_path=path, exists=path.exists())
        for role, path in REVIEWER_ROLE_PACKETS.items()
    ]


def _missing_template_paths() -> list[str]:
    return [str(p).replace("\\", "/") for p in REQUIRED_TEMPLATE_PATHS if not p.exists()]


def _next_best_reviewer() -> str:
    # Order chosen to match the 10/10-under-constraints roadmap's
    # B-tier priorities.  Highest unlock for the medical side is the
    # oncology nurse / clinician review; held-out RAG author is the
    # highest unlock for RAG evaluation credibility.
    return "oncology_nurse_or_resident"


def build_readiness() -> dict[str, Any]:
    completed = _count_committed_attestations()
    prepared = _prepared_packets()
    missing_templates = _missing_template_paths()
    prepared_packet_paths = [str(p.packet_path).replace("\\", "/") for p in prepared if p.exists]
    missing_packet_paths = [str(p.packet_path).replace("\\", "/") for p in prepared if not p.exists]

    if completed > 0:
        status = "in_progress" if completed < len(PENDING_REVIEW_TYPES) else "needs_attention"
    elif missing_templates or missing_packet_paths:
        status = "needs_attention"
    else:
        status = "ready_to_request_review"

    return {
        "schema_version": "external_review_execution_readiness_v1",
        "status": status,
        "label": "external_review_execution_readiness",
        "clinical_validation": False,
        "claim_boundary": (
            "Reviewer execution readiness — engineering scaffolding only.  "
            "Counts attestation files actually committed in "
            "Data/evals/external_review/.  Does NOT fabricate, infer, or "
            "imply completed reviews under any circumstance.  Not clinical "
            "validation, not clinician sign-off, not IRB clearance, not "
            "approval."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "completed_reviews": completed,
        "pending_review_types": list(PENDING_REVIEW_TYPES),
        "prepared_templates": [str(p).replace("\\", "/") for p in REQUIRED_TEMPLATE_PATHS if p.exists()],
        "missing_templates": missing_templates,
        "prepared_packets": prepared_packet_paths,
        "missing_packets": missing_packet_paths,
        "reviewer_roles_needed": sorted(REVIEWER_ROLE_PACKETS.keys()),
        "next_best_reviewer": _next_best_reviewer(),
        "anti_fabrication_invariant": (
            "completed_reviews is computed only from real *_attestation.md "
            "files (excluding the template) with a ticked boundary "
            "acknowledgement checkbox.  An empty or template-shaped file "
            "MUST NOT increment this count."
        ),
    }


def write_readiness(output_path: Path = OUTPUT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(build_readiness(), indent=2), encoding="utf-8")
    return output_path


__all__ = [
    "OUTPUT_PATH",
    "PENDING_REVIEW_TYPES",
    "REQUIRED_TEMPLATE_PATHS",
    "REVIEWER_ROLE_PACKETS",
    "build_readiness",
    "write_readiness",
]
