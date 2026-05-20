from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.local_llm import configured_llm_providers
from backend.services.oncology_canonical_schema import ROOT_DIR


DEFAULT_OUTPUT_PATH = "Data/evals/ops/latest_local_slm_readiness.json"

ALLOWED_LOCAL_SLM_TASKS = {
    "intent_classification": "Low-risk routing hint only; deterministic safety gates still run first.",
    "query_rewriting": "Rewrite/retrieve helper only; source-governed RAG still controls answer context.",
    "claim_extraction": "Extract candidate claims for validator review; cannot approve claims.",
    "summary_formatting": "Formatting only; cannot introduce medical content.",
    "portal_help": "Allowed for navigation/help copy when no patient-specific medical advice is involved.",
    "refusal_style_drafting": "May draft tone for an already-decided refusal; validators keep final authority.",
}

BLOCKED_LOCAL_SLM_SOLO_TASKS = {
    "diagnosis",
    "treatment_advice",
    "prognosis",
    "genetic_risk_interpretation",
    "tumor_marker_interpretation",
    "medication_safety",
    "supplement_safety",
    "dosage_guidance",
}

REQUIRED_GATES_AFTER_LOCAL_SLM = [
    "deterministic_pre_generation_safety_gate",
    "source_governance_for_any_rag_context",
    "medical_claim_boundary_checker",
    "claim_level_citation_validation_for_patient_education",
    "post_generation_safety_validator",
    "release_gate_artifact_checks",
]

CLAIM_BOUNDARY = (
    "Local SLM readiness is an engineering routing/cost scaffold. It is not "
    "clinical validation and does not authorize local models to answer medical "
    "decision questions without the existing guardrails."
)


def is_local_slm_task_allowed(task: str) -> bool:
    normalized = _normalize_task(task)
    return normalized in ALLOWED_LOCAL_SLM_TASKS and normalized not in BLOCKED_LOCAL_SLM_SOLO_TASKS


def build_local_slm_readiness_manifest(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    providers = configured_llm_providers()
    payload = {
        "schema_version": "local_slm_readiness_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong",
        "summary": {
            "enabled_low_risk_task_count": len(ALLOWED_LOCAL_SLM_TASKS),
            "blocked_solo_task_count": len(BLOCKED_LOCAL_SLM_SOLO_TASKS),
            "production_default": "disabled_or_optional_helper_only",
        },
        "providers_detected": providers,
        "enabled_low_risk_tasks": [
            {"task": task, "allowed": True, "scope": scope}
            for task, scope in sorted(ALLOWED_LOCAL_SLM_TASKS.items())
        ],
        "blocked_solo_tasks": [
            {
                "task": task,
                "allowed": False,
                "reason": "Patient-specific medical decision or high-risk clinical interpretation.",
            }
            for task in sorted(BLOCKED_LOCAL_SLM_SOLO_TASKS)
        ],
        "required_gates_after_local_slm": REQUIRED_GATES_AFTER_LOCAL_SLM,
        "route_policy": {
            "local_slm_may_reduce_cost_for": sorted(ALLOWED_LOCAL_SLM_TASKS),
            "local_slm_must_not_be_final_authority_for": sorted(BLOCKED_LOCAL_SLM_SOLO_TASKS),
            "production_default": "disabled_or_optional_helper_only",
        },
        "release_gate_expectations": [
            "unsafe_answer_rate must remain 0",
            "source_tier_correctness must remain 1.0 for patient-facing RAG",
            "claim validators must pass even when local SLM is used upstream",
            "blocked claim categories must still be refused deterministically",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(output_path, payload)
    return payload


def _normalize_task(task: str) -> str:
    return str(task or "").strip().lower().replace("-", "_").replace(" ", "_")


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    candidate = Path(path)
    full_path = candidate if candidate.is_absolute() else ROOT_DIR / candidate
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = [
    "ALLOWED_LOCAL_SLM_TASKS",
    "BLOCKED_LOCAL_SLM_SOLO_TASKS",
    "DEFAULT_OUTPUT_PATH",
    "build_local_slm_readiness_manifest",
    "is_local_slm_task_allowed",
]
