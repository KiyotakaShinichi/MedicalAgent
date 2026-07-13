from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.oncology_canonical_schema import ROOT_DIR


DEFAULT_OUTPUT_PATH = "Data/evals/governance/latest_platform_control_plane_architecture.json"
DEFAULT_DOC_PATH = "docs/platform_control_plane_architecture.md"

CLAIM_BOUNDARY = (
    "Platform control-plane architecture is an engineering roadmap and contract artifact. It does not establish "
    "clinical validation, real patient safety, patient benefit, clinician approval, IRB approval, HIPAA compliance, "
    "FHIR interoperability, or production healthcare readiness. It must not be used to claim diagnosis, prognosis, "
    "treatment recommendation, medication advice, genetic-risk interpretation, tumor-marker interpretation, or "
    "clinical workflow reduction."
)


def build_platform_control_plane_architecture(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    doc_path: str | Path = DEFAULT_DOC_PATH,
) -> dict[str, Any]:
    payload = {
        "schema_version": "platform_control_plane_architecture_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong",
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "live_patient_authority_added": False,
        "sections": {
            "agent_state_machine": _agent_state_machine(),
            "rag_control_plane": _rag_control_plane(),
            "medical_policy_registry": _medical_policy_registry(),
            "ml_feature_store_schema": _ml_feature_store_schema(),
            "eval_ops_registry": _eval_ops_registry(),
            "trace_envelope_v2": _trace_envelope_v2(),
            "background_eval_worker": _background_eval_worker(),
            "integration_boundaries": _integration_boundaries(),
            "implementation_sequence": _implementation_sequence(),
        },
        "acceptance_checks": [
            "agent transitions have explicit reason, safety_level, latency_ms, and blocked_alternatives fields",
            "medical policy registry blocks diagnosis/treatment/prognosis/genetics/tumor-marker authority",
            "RAG controller preserves source-tier and allowed-use filters before generation",
            "ML feature store keeps synthetic features versioned and non-promotional",
            "eval registry separates blocker, warning, supporting, and informational artifacts",
            "trace envelope stores decision metadata only and excludes private chain-of-thought",
            "background worker runs eval/admin jobs only and cannot execute clinical actions",
            "n8n and Pinecone remain optional, disabled by default, and PHI-blocked",
        ],
        "blocked_claims": [
            "clinical validation",
            "real-world safety guarantee",
            "patient benefit",
            "clinician approval",
            "IRB approval",
            "HIPAA compliance",
            "FHIR interoperability",
            "production healthcare readiness",
            "diagnostic authority",
            "treatment recommendation",
            "prognosis or survival estimate",
            "genetic-risk interpretation",
            "tumor-marker interpretation",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(_resolve(output_path), payload)
    _write_doc(_resolve(doc_path), payload)
    return payload


def _agent_state_machine() -> dict[str, Any]:
    states = [
        "input_received",
        "safety_precheck",
        "intent_classified",
        "clarification_or_tool_planning",
        "retrieval_planning",
        "evidence_retrieved",
        "evidence_sufficiency_checked",
        "generation_or_refusal_selected",
        "post_generation_validated",
        "final_response_ready",
        "trace_persisted",
    ]
    transitions = [
        ["input_received", "safety_precheck"],
        ["safety_precheck", "intent_classified"],
        ["safety_precheck", "generation_or_refusal_selected"],
        ["intent_classified", "clarification_or_tool_planning"],
        ["intent_classified", "retrieval_planning"],
        ["clarification_or_tool_planning", "final_response_ready"],
        ["retrieval_planning", "evidence_retrieved"],
        ["evidence_retrieved", "evidence_sufficiency_checked"],
        ["evidence_sufficiency_checked", "retrieval_planning"],
        ["evidence_sufficiency_checked", "generation_or_refusal_selected"],
        ["generation_or_refusal_selected", "post_generation_validated"],
        ["post_generation_validated", "final_response_ready"],
        ["final_response_ready", "trace_persisted"],
    ]
    return {
        "status": "contract_ready_not_live_refactor",
        "states": states,
        "terminal_state": "trace_persisted",
        "transitions": [
            {
                "from": src,
                "to": dst,
                "required_trace_fields": [
                    "reason",
                    "safety_level",
                    "policy_decision",
                    "latency_ms",
                    "blocked_alternatives",
                ],
            }
            for src, dst in transitions
        ],
        "forbidden_transitions": [
            "evidence_retrieved -> final_response_ready without evidence_sufficiency_checked",
            "generation_or_refusal_selected -> final_response_ready without post_generation_validated",
            "tool_write -> final_response_ready without confirmation",
        ],
        "live_agent_change": False,
    }


def _rag_control_plane() -> dict[str, Any]:
    return {
        "status": "scaffolded_with_existing_eval_artifacts",
        "controller_steps": [
            "intent_specific_rag_mode_selection",
            "retrieval_backend_selection",
            "query_rewrite_or_decomposition",
            "local_faiss_bm25_or_pinecone_shadow_retrieval",
            "parent_child_context_expansion",
            "source_tier_allowed_use_filter",
            "evidence_sufficiency_grading",
            "conflict_detection",
            "citation_window_selection",
            "claim_source_alignment",
            "post_generation_validation",
        ],
        "backends": {
            "local_faiss_bm25": "primary",
            "pinecone": "shadow_only_disabled_by_default",
        },
        "promotion_rule": (
            "No retrieval backend may replace the local path unless source_tier_correctness remains 1.0, "
            "unsafe_answer_rate remains 0.0, citation precision does not regress, unsupported context does not "
            "increase, and latency/cost are reported."
        ),
        "clinical_validation": False,
    }


def _medical_policy_registry() -> dict[str, Any]:
    policies = [
        ("diagnosis_boundary", "blocks diagnosis and diagnosis confirmation"),
        ("treatment_change_boundary", "blocks start/stop/switch/delay treatment instructions"),
        ("dosage_boundary", "blocks medication dosing and dose adjustment"),
        ("prognosis_boundary", "blocks survival or recurrence prediction"),
        ("genetic_vus_boundary", "blocks genetic-risk interpretation and VUS-as-positive claims"),
        ("tumor_marker_boundary", "blocks tumor-marker conclusions such as recurrence proof"),
        ("supplement_pharmacist_boundary", "routes supplement safety/replacement questions to pharmacist or clinician review"),
        ("urgent_symptom_escalation", "routes urgent symptoms to appropriate urgent-care language"),
        ("emotional_distress_support", "allows empathic support without clinical authority"),
        ("privacy_cross_patient_boundary", "blocks PII requests and cross-patient exfiltration"),
        ("prompt_injection_boundary", "blocks instruction override and tool-leak attempts"),
    ]
    return {
        "status": "policy_as_code_contract",
        "policies": [
            {
                "policy_id": policy_id,
                "purpose": purpose,
                "allowed_response_types": [
                    "education",
                    "record_organization",
                    "missing_data_explanation",
                    "review_routing",
                    "safe_refusal",
                    "urgent_escalation",
                    "questions_for_care_team",
                ],
                "blocked_response_types": [
                    "diagnosis",
                    "treatment_recommendation",
                    "dosage_change",
                    "prognosis",
                    "genetic_risk_prediction",
                    "tumor_marker_conclusion",
                    "false_reassurance",
                ],
                "requires_test_cases": True,
            }
            for policy_id, purpose in policies
        ],
        "clinical_validation": False,
    }


def _ml_feature_store_schema() -> dict[str, Any]:
    return {
        "status": "versioned_schema_contract",
        "feature_store_name": "synthetic_temporal_monitoring_feature_store",
        "entities": ["synthetic_patient", "treatment_cycle", "modality_snapshot", "prediction_head"],
        "feature_groups": {
            "demographics_context": ["age", "synthetic_subtype", "stage_context"],
            "cbc_labs": ["pre_wbc", "nadir_wbc", "hemoglobin", "platelets", "anc_proxy", "missingness_flags"],
            "symptoms": ["symptom_count", "max_symptom_severity", "persistence_proxy"],
            "imaging_context": ["mri_percent_change", "imaging_available", "report_trend_context"],
            "treatment_context": ["cycle_index", "dose_delay_context", "modality_combination_context"],
            "biomarker_context": ["er_status", "pr_status", "her2_status", "ki67_context"],
            "governance": ["lineage_hash", "schema_version", "generator_seed", "split_id"],
        },
        "prediction_heads": [
            {"head": "response_classification", "safe_use": "monitor_only"},
            {"head": "response_score_regression", "safe_use": "monitor_only"},
            {"head": "toxicity_review_signal", "safe_use": "review_hint_only"},
            {"head": "genetic_biomarker_context", "safe_use": "context_only"},
            {"head": "tumor_marker_context", "safe_use": "context_only"},
        ],
        "promotion_block": "No head may influence treatment decisions under current synthetic-only evidence.",
        "clinical_validation": False,
    }


def _eval_ops_registry() -> dict[str, Any]:
    return {
        "status": "registry_contract",
        "artifact_tiers": {
            "hard_blocker": [
                "unsafe leakage on critical routes",
                "medical claim-boundary regression",
                "leakage audit failure",
                "patient-facing clinical overclaim",
            ],
            "warning": [
                "heldout adversarial below target",
                "retrieval improvement not proven",
                "normal RAG p95 above budget",
                "unsupported_context_rate too high",
                "trace coverage needs_attention",
            ],
            "supporting": [
                "source-tier ablation",
                "synthetic data quality proxy",
                "external dataset readiness map",
                "n8n/Pinecone optional scaffold",
            ],
            "informational": [
                "unreviewed packets",
                "future access packets",
                "negative-results gallery",
                "not-completed holdouts",
            ],
        },
        "required_metadata": [
            "artifact_path",
            "owner",
            "tier",
            "status",
            "last_run_at",
            "max_age_days",
            "contamination_status",
            "clinical_validation",
            "strongest_allowed_reading",
            "known_limitations",
            "next_action",
        ],
        "clinical_validation": False,
    }


def _trace_envelope_v2() -> dict[str, Any]:
    return {
        "status": "contract_ready",
        "required_fields": [
            "request_id",
            "correlation_id",
            "patient_id_hash",
            "route",
            "intent",
            "safety_decision",
            "policy_decision",
            "retrieval_backend",
            "source_ids",
            "claim_validation",
            "post_generation_decision",
            "cache_status",
            "latency_ms",
            "estimated_cost",
            "final_policy_status",
        ],
        "forbidden_fields": [
            "private_chain_of_thought",
            "raw_prompt_with_secrets",
            "unredacted_phi",
            "raw_patient_identifier",
        ],
        "storage_rule": "Store decision metadata and evidence references only.",
        "clinical_validation": False,
    }


def _background_eval_worker() -> dict[str, Any]:
    return {
        "status": "worker_contract",
        "allowed_job_types": [
            "refresh_non_live_eval_artifact",
            "run_release_gate",
            "generate_negative_results_gallery",
            "create_stale_artifact_ticket",
            "prepare_reviewer_packet_reminder",
            "run_pinecone_shadow_dry_run",
        ],
        "blocked_job_types": [
            "diagnosis",
            "treatment_recommendation",
            "dosage_change",
            "prognosis",
            "clinical_escalation_without_human_review",
            "send_phi_to_external_service",
        ],
        "queue_contract": {
            "job_id": "uuid",
            "job_type": "enum",
            "requested_by": "admin/system",
            "payload_redacted": True,
            "timeout_seconds": 900,
            "retry_policy": "max_2_with_backoff",
        },
        "clinical_validation": False,
    }


def _integration_boundaries() -> dict[str, Any]:
    return {
        "n8n": {
            "status": "internal_automation_only",
            "allowed": ["release alerts", "stale artifact tickets", "reviewer reminders", "eval refresh triggers"],
            "blocked": ["PHI workflows", "patient-facing clinical advice", "clinical action automation"],
        },
        "pinecone": {
            "status": "shadow_retrieval_only",
            "allowed": ["synthetic/demo KB retrieval comparison", "metadata-filter stress testing"],
            "blocked": ["raw patient chat storage", "patient memory", "clinical confidence score"],
        },
        "clinical_validation": False,
    }


def _implementation_sequence() -> list[dict[str, Any]]:
    return [
        {"rank": 1, "task": "standardize trace envelope v2 across agent routes", "risk": "medium"},
        {"rank": 2, "task": "implement agent state-machine event emitter around existing branches", "risk": "medium"},
        {"rank": 3, "task": "generate policy-registry tests from medical boundary registry", "risk": "low"},
        {"rank": 4, "task": "create eval ops registry endpoint for admin dashboard", "risk": "low"},
        {"rank": 5, "task": "version synthetic feature-store schema and row lineage manifests", "risk": "low"},
        {"rank": 6, "task": "add background eval worker with admin-only jobs", "risk": "medium"},
        {"rank": 7, "task": "run Pinecone shadow comparison only when credentials are explicitly configured", "risk": "medium"},
        {"rank": 8, "task": "wire n8n workflow import only for redacted admin events", "risk": "medium"},
    ]


def _write_doc(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sections = payload["sections"]
    lines = [
        "# Platform Control Plane Architecture",
        "",
        payload["claim_boundary"],
        "",
        "## Agent State Machine",
        "",
        f"- Status: `{sections['agent_state_machine']['status']}`",
        f"- States: `{len(sections['agent_state_machine']['states'])}`",
        f"- Transitions: `{len(sections['agent_state_machine']['transitions'])}`",
        "",
        "## RAG Control Plane",
        "",
        *[f"- {step}" for step in sections["rag_control_plane"]["controller_steps"]],
        "",
        "## Medical Policy Registry",
        "",
        *[
            f"- `{policy['policy_id']}`: {policy['purpose']}"
            for policy in sections["medical_policy_registry"]["policies"]
        ],
        "",
        "## ML Feature Store Schema",
        "",
        *[f"- `{group}`: {', '.join(fields)}" for group, fields in sections["ml_feature_store_schema"]["feature_groups"].items()],
        "",
        "## Eval Ops Registry",
        "",
        *[f"- `{tier}`: {', '.join(items)}" for tier, items in sections["eval_ops_registry"]["artifact_tiers"].items()],
        "",
        "## Trace Envelope V2",
        "",
        *[f"- `{field}`" for field in sections["trace_envelope_v2"]["required_fields"]],
        "",
        "## Background Eval Worker",
        "",
        "Allowed jobs:",
        *[f"- {job}" for job in sections["background_eval_worker"]["allowed_job_types"]],
        "",
        "Blocked jobs:",
        *[f"- {job}" for job in sections["background_eval_worker"]["blocked_job_types"]],
        "",
        "## Implementation Sequence",
        "",
        *[f"{item['rank']}. {item['task']} (risk: {item['risk']})" for item in sections["implementation_sequence"]],
        "",
        "## Blocked Claims",
        "",
        *[f"- {claim}" for claim in payload["blocked_claims"]],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate


__all__ = [
    "CLAIM_BOUNDARY",
    "DEFAULT_DOC_PATH",
    "DEFAULT_OUTPUT_PATH",
    "build_platform_control_plane_architecture",
]
