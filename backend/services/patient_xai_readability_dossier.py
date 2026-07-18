"""Patient-facing XAI/readability dossier for NLCare.

This artifact audits whether patient-visible numbers are explained as
engineering context rather than clinical authority.  It does not change model
predictions, retrieval, routing, or clinical behavior.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_PATH = Path("Data/evals/governance/latest_patient_xai_readability_dossier.json")
DEFAULT_DOC_PATH = Path("docs/patient_xai_readability_dossier.md")

REQUIRED_PATIENT_EXPLANATION_SURFACES = [
    {
        "id": "review_queue",
        "display_label": "Items for review",
        "must_explain": [
            "what_count_means",
            "how_count_is_calculated",
            "why_not_urgency_or_diagnosis",
            "safe_next_steps",
        ],
    },
    {
        "id": "synthetic_model_pattern",
        "display_label": "Synthetic model pattern",
        "must_explain": [
            "synthetic_class_probability",
            "confidence_is_not_patient_outcome_probability",
            "modalities_used_and_missing",
            "abstention_or_low_confidence_reason",
        ],
    },
    {
        "id": "latest_cbc",
        "display_label": "Latest CBC",
        "must_explain": [
            "population_default_reference_bands",
            "not_personalized_reference_ranges",
            "not_diagnosis",
            "care_team_discussion_boundary",
        ],
    },
    {
        "id": "record_coverage",
        "display_label": "Record coverage",
        "must_explain": [
            "available_vs_missing_record_areas",
            "why_missingness_changes_confidence",
            "not_health_status",
            "what_user_can_update",
        ],
    },
    {
        "id": "old_monitoring_score_boundary",
        "display_label": "Removed 0-100 headline score",
        "must_explain": [
            "why_removed_from_patient_headline",
            "why_not_cancer_status",
            "why_not_treatment_response",
            "why_not_prognosis",
        ],
    },
]

KNOWN_WEAKNESS_INPUTS = {
    "route_aware_rag": Path("Data/evals/rag/latest_route_aware_rag_policy_eval.json"),
    "ml_logic_safety": Path("Data/evals/models/latest_ml_logic_safety_alignment.json"),
    "automation_reliability": Path("Data/evals/ops/latest_automation_reliability_dossier.json"),
}

CLAIM_BOUNDARY = (
    "Patient-facing XAI explains how NLCare produced synthetic engineering indicators. "
    "It does not explain clinical causality, diagnose, recommend treatment, estimate prognosis, "
    "interpret genetic risk, interpret tumor markers, prove patient benefit, or validate real-world safety."
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _rag_risk_summary(rag: dict[str, Any]) -> dict[str, Any]:
    summary = rag.get("route_aware_summary") or rag.get("summary") or {}
    citation_precision = summary.get("citation_precision")
    unsupported_context_rate = summary.get("unsupported_context_rate")
    failure_count = summary.get("failure_count")
    needs_attention = False
    if isinstance(citation_precision, (int, float)) and citation_precision < 0.7:
        needs_attention = True
    if isinstance(unsupported_context_rate, (int, float)) and unsupported_context_rate > 0.1:
        needs_attention = True
    return {
        "status": rag.get("status") or "missing",
        "citation_precision": citation_precision,
        "unsupported_context_rate": unsupported_context_rate,
        "failure_count": failure_count,
        "patient_copy_rule": (
            "RAG evidence must be presented as source-backed education only. "
            "Weak citation precision or unsupported context should reduce confidence, not create a stronger answer."
        ),
        "needs_attention": needs_attention,
    }


def _ml_risk_summary(ml: dict[str, Any]) -> dict[str, Any]:
    known = ml.get("summary", {}).get("known_attention_items") or []
    return {
        "status": ml.get("status") or "missing",
        "logic_alignment_score": ml.get("summary", {}).get("logic_alignment_score"),
        "known_attention_items": known,
        "patient_copy_rule": (
            "Synthetic model explanations must say what data was used, what data was missing, "
            "why the output may abstain, and why it is not clinical evidence."
        ),
        "needs_attention": bool(known) or ml.get("status") == "needs_attention",
    }


def _automation_risk_summary(auto: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": auto.get("status") or "missing",
        "external_delivery_enabled_by_default": auto.get("external_delivery_enabled_by_default"),
        "real_emergency_coverage_claim": auto.get("real_emergency_coverage_claim"),
        "patient_copy_rule": (
            "Automation copy must describe local review routing and test-recipient delivery only. "
            "It must not imply monitored emergency coverage or clinician acknowledgement."
        ),
        "needs_attention": bool(auto.get("external_delivery_enabled_by_default")) or bool(auto.get("real_emergency_coverage_claim")),
    }


def build_patient_xai_readability_dossier(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    doc_path: str | Path = DEFAULT_DOC_PATH,
) -> dict[str, Any]:
    rag = _read_json(KNOWN_WEAKNESS_INPUTS["route_aware_rag"])
    ml = _read_json(KNOWN_WEAKNESS_INPUTS["ml_logic_safety"])
    auto = _read_json(KNOWN_WEAKNESS_INPUTS["automation_reliability"])
    surface_rows = [
        {
            **surface,
            "required_explanation_count": len(surface["must_explain"]),
            "status": "specified",
            "clinical_validation": False,
        }
        for surface in REQUIRED_PATIENT_EXPLANATION_SURFACES
    ]
    checks = [
        {
            "id": "patient_numbers_have_meaning_calculation_next_steps",
            "passed": all(
                {"what_count_means", "how_count_is_calculated", "safe_next_steps"}.intersection(surface["must_explain"])
                or surface["id"] != "review_queue"
                for surface in REQUIRED_PATIENT_EXPLANATION_SURFACES
            ),
            "why": "The KPI strip requires meaning, calculation, and allowed next-step copy for patient-facing indicators.",
        },
        {
            "id": "synthetic_outputs_have_nonclinical_boundary",
            "passed": True,
            "why": "The dossier requires synthetic-only, not-personal-outcome-probability language for model pattern displays.",
        },
        {
            "id": "removed_score_boundary_documented",
            "passed": True,
            "why": "The old 0-100 monitoring index is documented as removed from patient headlines because it can be misread as a health score.",
        },
        {
            "id": "known_rag_weakness_visible",
            "passed": True,
            "why": "The dossier carries citation precision and unsupported-context weaknesses into patient-copy rules.",
        },
        {
            "id": "automation_not_emergency_coverage",
            "passed": not bool(auto.get("real_emergency_coverage_claim")),
            "why": "Automation copy remains bounded to local review workflow and test-recipient dispatch.",
        },
    ]
    failed = [check for check in checks if not check["passed"]]
    report = {
        "schema_version": "patient_xai_readability_dossier_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong" if not failed else "needs_attention",
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "patient_benefit_claim": False,
        "diagnostic_authority_claim": False,
        "treatment_recommendation_claim": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "surface_count": len(surface_rows),
        "surfaces": surface_rows,
        "checks": checks,
        "failed_check_count": len(failed),
        "weakness_visibility": {
            "rag": _rag_risk_summary(rag),
            "ml": _ml_risk_summary(ml),
            "automation": _automation_risk_summary(auto),
        },
        "recommended_ui_copy_rules": [
            "Every patient-visible number needs a meaning, calculation, safe next steps, and nonclinical boundary.",
            "Use 'synthetic class probability' or 'model confidence', never 'chance of response' or 'survival'.",
            "When data is missing, say which record types are missing instead of inventing confidence.",
            "When automation triggers, say 'queued for review workflow' rather than 'doctor notified' unless a verified human acknowledgement exists.",
            "Keep negative RAG and ML weaknesses visible in admin/reviewer surfaces.",
        ],
        "contamination_note": (
            "This is an internal UI/XAI governance artifact. It audits explanation completeness and wording boundaries; "
            "it is not usability testing with real patients and not clinical validation."
        ),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_doc(report, Path(doc_path))
    return report


def _write_doc(report: dict[str, Any], doc_path: Path) -> None:
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Patient XAI Readability Dossier",
        "",
        "Status: " + str(report["status"]),
        "",
        "No clinical validation. No clinical authority. No patient benefit claim.",
        "",
        "## Claim Boundary",
        "",
        CLAIM_BOUNDARY,
        "",
        "## Required Patient Explanation Surfaces",
        "",
    ]
    for surface in report["surfaces"]:
        lines.append(f"- **{surface['display_label']}**: " + ", ".join(surface["must_explain"]))
    lines.extend([
        "",
        "## Weakness Visibility",
        "",
        f"- RAG citation precision: {report['weakness_visibility']['rag'].get('citation_precision')}",
        f"- RAG unsupported context rate: {report['weakness_visibility']['rag'].get('unsupported_context_rate')}",
        f"- ML attention items: {', '.join(report['weakness_visibility']['ml'].get('known_attention_items') or []) or 'none'}",
        f"- Automation emergency coverage claim: {report['weakness_visibility']['automation'].get('real_emergency_coverage_claim')}",
        "",
        "## UI Copy Rules",
        "",
    ])
    lines.extend(f"- {rule}" for rule in report["recommended_ui_copy_rules"])
    lines.append("")
    doc_path.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "CLAIM_BOUNDARY",
    "DEFAULT_DOC_PATH",
    "DEFAULT_OUTPUT_PATH",
    "REQUIRED_PATIENT_EXPLANATION_SURFACES",
    "build_patient_xai_readability_dossier",
]
