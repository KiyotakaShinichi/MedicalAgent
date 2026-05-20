from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from backend.services.oncology_canonical_schema import ROOT_DIR


EVENT_TYPES = (
    "safety_gate_triggered",
    "rag_retrieval_started",
    "rag_retrieval_completed",
    "source_filtering_completed",
    "claim_validation_completed",
    "post_generation_validator_triggered",
    "model_prediction_generated",
    "abstention_generated",
    "clinician_review_action",
    "artifact_generated",
    "release_gate_pass",
    "release_gate_fail",
    "ab_candidate_rejected",
    "medical_claim_boundary_triggered",
)

SEVERITIES = ("debug", "info", "warning", "error", "critical")


class EventRecord(BaseModel):
    event_type: str = Field(pattern="|".join(EVENT_TYPES))
    severity: str = Field(pattern="|".join(SEVERITIES))
    request_id: str
    user_role: str
    timestamp: str
    trace_id: str
    patient_id: str | None = None
    artifact_id: str | None = None
    model_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def build_event(
    *,
    event_type: str,
    severity: str = "info",
    request_id: str = "local",
    user_role: str = "system",
    trace_id: str = "local-trace",
    patient_id: str | None = None,
    artifact_id: str | None = None,
    model_version: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return EventRecord(
        event_type=event_type,
        severity=severity,
        request_id=request_id,
        user_role=user_role,
        timestamp=datetime.now(timezone.utc).isoformat(),
        trace_id=trace_id,
        patient_id=patient_id,
        artifact_id=artifact_id,
        model_version=model_version,
        metadata=metadata or {},
    ).model_dump()


def write_event_taxonomy_doc(
    *,
    output_path: str = "Data/evals/ops/latest_event_taxonomy_manifest.json",
    doc_path: str = "docs/event_taxonomy.md",
) -> dict[str, Any]:
    payload = {
        "schema_version": "event_taxonomy_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong",
        "event_types": list(EVENT_TYPES),
        "severities": list(SEVERITIES),
        "required_fields": list(EventRecord.model_fields.keys()),
        "claim_boundary": "PoC event taxonomy for audit discipline; not production SRE or compliance monitoring.",
    }
    _write_json(_resolve(output_path), payload)
    lines = ["# Event Taxonomy", "", payload["claim_boundary"], "", "## Event Types"]
    lines.extend(f"- `{event}`" for event in EVENT_TYPES)
    _resolve(doc_path).parent.mkdir(parents=True, exist_ok=True)
    _resolve(doc_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate
