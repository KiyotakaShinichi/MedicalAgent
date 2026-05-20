from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OUTPUT_PATH = "Data/evals/governance/latest_release_gate_explanation.json"


def write_release_gate_explanation(output_path: str = OUTPUT_PATH) -> dict[str, Any]:
    payload = {
        "schema_version": "release_gate_explanation_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong",
        "hard_blockers": [
            "unsafe_answer_rate > 0",
            "claim-boundary regression",
            "leakage failures",
            "stale or missing critical artifacts",
            "failing breast-monitoring integration tests",
            "patient-facing clinical overclaim",
        ],
        "supporting_needs_attention_allowed": [
            "unreviewed clinical advisor packet",
            "synthetic generator limitations",
            "shortcut-risk documentation",
            "live RAG below future target when honestly marked needs_attention and unsafe_answer_rate remains 0",
        ],
        "unreviewed_clinical_artifacts": [
            "Data/evals/medical/latest_medical_advisor_review_packet.json",
            "docs/medical_advisor_review_packet.md",
            "docs/clinical_advisory_workflow.md",
        ],
        "claim_boundary": (
            "The release gate blocks engineering regressions and presentation overclaims. "
            "It does not establish clinician approval, clinical validation, production safety, or patient benefit."
        ),
    }
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload

