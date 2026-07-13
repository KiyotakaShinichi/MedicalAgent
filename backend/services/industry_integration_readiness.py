from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.oncology_canonical_schema import ROOT_DIR


DEFAULT_OUTPUT_PATH = "Data/evals/ops/latest_industry_integration_readiness.json"
DEFAULT_DOC_PATH = "docs/industry_integration_n8n_pinecone.md"

CLAIM_BOUNDARY = (
    "n8n and Pinecone integration readiness is software architecture planning only. It does not make NLCare "
    "clinically validated, HIPAA compliant, production healthcare ready, clinician-approved, or safe for real "
    "patient care. External workflow automation and managed vector search must remain optional and disabled "
    "for patient-specific or PHI workflows until compliance, security review, and clinical governance exist."
)


def build_industry_integration_readiness(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    doc_path: str | Path = DEFAULT_DOC_PATH,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "industry_integration_readiness_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready_for_optional_scaffold",
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "hipaa_compliance_claim": False,
        "live_patient_route_enabled": False,
        "phi_allowed": False,
        "recommended_order": [
            "n8n for internal evaluation/review automation only",
            "Pinecone shadow index for synthetic/demo KB retrieval only",
            "dual-run FAISS/BM25/Pinecone retrieval comparison",
            "optional admin-only retrieval diagnostics",
            "compliance/security review before any real patient data or PHI",
        ],
        "integrations": {
            "n8n": _n8n_plan(),
            "pinecone": _pinecone_plan(),
        },
        "deployment_env_contract": {
            "n8n_optional": {
                "N8N_BASE_URL": "Base URL for self-hosted or cloud n8n instance.",
                "NLC_N8N_WEBHOOK_SECRET": "Shared HMAC/signature secret for callbacks.",
                "NLC_N8N_EVAL_WEBHOOK_URL": "Optional eval-run webhook endpoint.",
                "NLC_N8N_REVIEW_WEBHOOK_URL": "Optional reviewer-intake webhook endpoint.",
            },
            "pinecone_optional": {
                "PINECONE_API_KEY": "Managed vector DB API key; never committed.",
                "PINECONE_INDEX_HOST": "Index host for SDK data-plane calls.",
                "PINECONE_NAMESPACE_KB": "Synthetic/demo knowledge-base namespace.",
                "PINECONE_ENABLED": "Must default to false.",
            },
        },
        "acceptance_checks_before_live_use": [
            "External services disabled by default in local and demo configs.",
            "No PHI or patient-specific chat turns sent to n8n or Pinecone.",
            "Pinecone retrieval must preserve same source-tier filtering, allowed-use filtering, staleness checks, and citation validation.",
            "n8n workflows may trigger evals, review intake, and admin alerts; they may not issue medical advice or treatment actions.",
            "All outbound requests carry request IDs and redact patient identifiers.",
            "Rate limiting, retry/backoff, timeout, and audit logging exist before shadow mode.",
            "A security/compliance review is required before real patient data.",
        ],
        "blocked_workflows": [
            "automatic diagnosis or triage decisions",
            "automatic treatment, dosage, medication, supplement, prognosis, genetics, or tumor-marker advice",
            "sending raw patient chat logs or PHI to external SaaS",
            "using Pinecone score as clinical confidence",
            "allowing n8n workflow output to bypass NLCare safety validators",
        ],
        "source_docs": [
            "https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.webhook/",
            "https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-langchain.chattrigger/",
            "https://docs.pinecone.io/guides/index-data/data-modeling",
            "https://docs.pinecone.io/guides/search/filter-by-metadata",
            "https://docs.pinecone.io/guides/index-data/upsert-data",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(_resolve(output_path), payload)
    _write_doc(_resolve(doc_path), payload)
    return payload


def _n8n_plan() -> dict[str, Any]:
    return {
        "role": "internal_workflow_automation",
        "status": "optional_disabled_by_default",
        "recommended_uses": [
            "release-gate result notification to Discord/Slack/email",
            "external reviewer intake and attestation reminders",
            "scheduled eval refresh workflow for non-live synthetic/internal artifacts",
            "admin-only incident ticket creation when unsafe leakage or stale blockers appear",
            "dataset integration checklist tracking for BreastDCEDL, Duke MRI, GENIE BPC, MIMIC-IV, and ClinVar",
        ],
        "not_allowed_uses": [
            "patient-facing clinical advice",
            "automatic clinical escalation without human review",
            "treatment or dosage decisions",
            "genetic counseling or VUS interpretation",
            "tumor-marker conclusion workflow",
            "PHI workflow before compliance review",
        ],
        "technical_pattern": [
            "FastAPI emits signed admin event to n8n Webhook node.",
            "n8n workflow performs notification, ticketing, or reviewer-intake task.",
            "n8n callback, if any, returns only workflow status to an admin endpoint.",
            "NLCare validators remain the final safety layer for any text shown in-app.",
        ],
        "security_requirements": [
            "HMAC or equivalent shared-secret verification for incoming/outgoing webhooks",
            "least-privilege workflow credentials",
            "no secrets in workflow JSON exports",
            "redacted payload schema by default",
            "separate test and production webhook URLs",
        ],
    }


def _pinecone_plan() -> dict[str, Any]:
    return {
        "role": "optional_managed_vector_backend_shadow_mode",
        "status": "optional_disabled_by_default",
        "recommended_uses": [
            "shadow retrieval comparison against FAISS/BM25 on synthetic/demo KB",
            "managed namespace experiments for source-tier governance",
            "metadata-filter stress testing for source_tier, allowed_use, patient_facing, and kb_fingerprint",
            "latency/cost comparison artifact before any promotion",
        ],
        "not_allowed_uses": [
            "raw patient chat or PHI storage",
            "replacement of source-tier filtering",
            "replacement of claim validation",
            "patient-specific memory before compliance review",
            "clinical confidence scoring",
        ],
        "namespace_plan": {
            "nlcare_kb_demo_t1_t3": "patient-facing synthetic/demo KB chunks only",
            "nlcare_eval_synthetic": "frozen eval chunks and synthetic test fixtures",
            "nlcare_clinician_only_shadow": "disabled by default; clinician-only docs never cited to patient-facing routes",
            "patient_data": "disallowed until compliance/security review",
        },
        "metadata_contract": [
            "source_id",
            "chunk_id",
            "source_tier",
            "allowed_use",
            "patient_facing",
            "staleness_status",
            "kb_fingerprint",
            "doc_type",
            "clinical_validation=false",
        ],
        "technical_pattern": [
            "Offline batch upsert of synthetic/demo KB chunks with full metadata.",
            "Shadow query Pinecone alongside local FAISS/BM25.",
            "Apply the same source-tier and allowed-use filters after retrieval.",
            "Compare recall, citation precision, unsupported context, source-tier correctness, latency, and cost.",
            "Keep local FAISS/BM25 as fallback until Pinecone beats or justifies the tradeoff under frozen evals.",
        ],
    }


def _write_doc(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n8n = payload["integrations"]["n8n"]
    pinecone = payload["integrations"]["pinecone"]
    lines = [
        "# n8n + Pinecone Industry Integration Readiness",
        "",
        payload["claim_boundary"],
        "",
        "## Recommended Order",
        "",
        *[f"{idx}. {step}" for idx, step in enumerate(payload["recommended_order"], start=1)],
        "",
        "## n8n",
        "",
        f"- Role: `{n8n['role']}`",
        f"- Status: `{n8n['status']}`",
        "",
        "Recommended uses:",
        *[f"- {item}" for item in n8n["recommended_uses"]],
        "",
        "Not allowed uses:",
        *[f"- {item}" for item in n8n["not_allowed_uses"]],
        "",
        "## Pinecone",
        "",
        f"- Role: `{pinecone['role']}`",
        f"- Status: `{pinecone['status']}`",
        "",
        "Recommended uses:",
        *[f"- {item}" for item in pinecone["recommended_uses"]],
        "",
        "Not allowed uses:",
        *[f"- {item}" for item in pinecone["not_allowed_uses"]],
        "",
        "Namespace plan:",
        *[f"- `{key}`: {value}" for key, value in pinecone["namespace_plan"].items()],
        "",
        "## Acceptance Checks Before Live Use",
        "",
        *[f"- {item}" for item in payload["acceptance_checks_before_live_use"]],
        "",
        "## Official Docs",
        "",
        *[f"- {url}" for url in payload["source_docs"]],
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
    "build_industry_integration_readiness",
]
