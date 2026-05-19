from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.oncology_canonical_schema import ROOT_DIR


DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_student_constraint_elevation_plan.json"
DEFAULT_DOC_PATH = "docs/student_constraint_elevation_plan.md"

CLAIM_BOUNDARY = (
    "This plan lists controllable engineering upgrades under student-accessible constraints. "
    "It is not clinical validation and does not replace clinician review or real-world outcome evidence."
)


def build_student_constraint_elevation_plan(
    *,
    output_path: str = DEFAULT_OUTPUT_PATH,
    doc_path: str = DEFAULT_DOC_PATH,
) -> dict[str, Any]:
    payload = {
        "schema_version": "student_constraint_elevation_plan_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong",
        "highest_leverage_next_steps": [
            {
                "rank": 1,
                "step": "External distribution alignment from exported cBioPortal rows",
                "why_it_elevates": "Moves beyond dataset-name mapping into actual public-row distribution checks.",
                "student_feasible": True,
                "proof_artifact": "Data/evals/models/latest_external_distribution_alignment.json",
            },
            {
                "rank": 2,
                "step": "Common-feature model transfer stress test",
                "why_it_elevates": "Train on one source and evaluate feature behavior on another without pretending labels match.",
                "student_feasible": True,
                "proof_artifact": "Data/evals/models/latest_common_feature_transfer_stress.json",
            },
            {
                "rank": 3,
                "step": "Synthetic realism candidate generator tuned against public distributions",
                "why_it_elevates": "Creates a separate candidate dataset with documented improvements and regressions.",
                "student_feasible": True,
                "proof_artifact": "Data/evals/models/latest_public_distribution_realism_candidate.json",
            },
            {
                "rank": 4,
                "step": "Human-review simulation packet with blinded rubric",
                "why_it_elevates": "Prepares for eventual nurse/clinician review while letting non-clinicians audit clarity and safety boundaries now.",
                "student_feasible": True,
                "proof_artifact": "future docs/reviewer_packet_blinded/",
            },
            {
                "rank": 5,
                "step": "Model behavior cards per head",
                "why_it_elevates": "Separates classification, regression, toxicity, genetics, and tumor-marker behavior so reviewers do not confuse scope.",
                "student_feasible": True,
                "proof_artifact": "future docs/model_behavior_cards/",
            },
            {
                "rank": 6,
                "step": "RAG answer provenance snapshots",
                "why_it_elevates": "Exports trace replay into compact reviewer-facing before/after examples.",
                "student_feasible": True,
                "proof_artifact": "future Data/evals/rag/latest_trace_replay_gallery.json",
            },
        ],
        "do_not_do_yet": [
            "Do not claim real-world response prediction.",
            "Do not promote toxicity target v2 beyond review-priority experiment.",
            "Do not use TCGA/METABRIC survival labels as if they were pCR or OncoTrack response-score labels.",
            "Do not train patient-facing treatment recommendations.",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(_resolve(output_path), payload)
    _write_doc(_resolve(doc_path), payload)
    return payload


def _write_doc(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Student-Constraint Elevation Plan",
        "",
        CLAIM_BOUNDARY,
        "",
        "## Highest Leverage Steps",
        "",
    ]
    for item in payload["highest_leverage_next_steps"]:
        lines.extend([
            f"{item['rank']}. **{item['step']}**",
            f"   - Why: {item['why_it_elevates']}",
            f"   - Proof artifact: `{item['proof_artifact']}`",
            "",
        ])
    lines.extend(["## Do Not Do Yet", ""])
    lines.extend(f"- {item}" for item in payload["do_not_do_yet"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate
