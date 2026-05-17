"""Consolidated medical-safety contract artifact."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.clinical_ontology import ontology_manifest
from backend.services.breast_cancer_journey import journey_phase_manifest
from backend.services.medical_claim_boundary import claim_boundary_manifest
from backend.services.medical_evidence_standards import standards_manifest


DEFAULT_OUTPUT_PATH = "Data/evals/safety/latest_medical_safety_contract.json"


def build_medical_safety_contract(
    *,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    payload = {
        "schema_version": "medical_safety_contract_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong",
        "clinical_ontology": ontology_manifest(),
        "minimum_evidence_standards": standards_manifest(),
        "medical_claim_boundary": claim_boundary_manifest(),
        "journey_phase_model": journey_phase_manifest(),
        "claim_boundary": (
            "This artifact documents engineering guardrails for allowed data "
            "values, minimum evidence, and blocked medical claim types. It is "
            "not clinical validation and is not a medical policy substitute."
        ),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_medical_safety_contract(path: str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {
            "schema_version": "medical_safety_contract_v1",
            "status": "missing",
            "message": "Run scripts/build_medical_safety_contract.py to generate the artifact.",
        }
    return json.loads(file_path.read_text(encoding="utf-8"))


__all__ = [
    "DEFAULT_OUTPUT_PATH",
    "build_medical_safety_contract",
    "load_medical_safety_contract",
]
