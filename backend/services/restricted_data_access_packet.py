from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.oncology_canonical_schema import ROOT_DIR


DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_restricted_data_access_packet.json"
DEFAULT_MD_PATH = "docs/data_access_packet/restricted_dataset_access_packet.md"

CLAIM_BOUNDARY = (
    "This packet prepares future data-access requests. It does not mean access has been granted, data have been "
    "received, or clinical validation has been performed."
)


DATASET_PACKETS: list[dict[str, Any]] = [
    {
        "dataset": "AACR GENIE BPC Breast Cancer",
        "priority": "highest_future_value",
        "why_it_matters": "real-world biomarker/genomic, treatment, and outcome context that could test treatment-context features",
        "fields_requested": [
            "patient/case identifier",
            "diagnosis and stage",
            "ER/PR/HER2 and molecular subtype",
            "genomic alterations and test type",
            "cancer-directed drug regimen history",
            "HER2-directed therapy history",
            "response/progression endpoints",
            "overall survival or follow-up outcome fields",
        ],
        "analysis_plan": [
            "map permitted fields into canonical ontology",
            "run leakage checks before modeling",
            "compare common-feature A/B candidates",
            "report failure cases and subgroup calibration",
            "keep all outputs non-diagnostic and non-treatment-recommending",
        ],
        "not_requested": ["free-text notes", "direct identifiers", "treatment recommendation permissions"],
    },
    {
        "dataset": "SEER breast registry",
        "priority": "population_distribution_check",
        "why_it_matters": "stage, subtype, surgery, radiation, and coarse treatment distribution checks",
        "fields_requested": [
            "age group",
            "stage",
            "ER/PR/HER2",
            "surgery",
            "radiation",
            "chemotherapy indicator where available",
            "survival/follow-up fields",
        ],
        "analysis_plan": [
            "compare synthetic cohort distributions against registry priors",
            "do not train NLCare response models from coarse treatment indicators alone",
        ],
        "not_requested": ["patient identifiers", "clinical notes", "full treatment-regimen inference"],
    },
    {
        "dataset": "SEER-Medicare",
        "priority": "future_claims_treatment_sequence_check",
        "why_it_matters": "claims can support richer surgery/radiation/chemo/endocrine/HER2-targeted sequence context in older patients",
        "fields_requested": [
            "diagnosis and staging variables",
            "procedure and treatment claims",
            "drug claims for chemotherapy/endocrine/HER2-targeted agents",
            "radiation claims",
            "follow-up/utilization outcomes",
        ],
        "analysis_plan": [
            "construct coarse treatment sequence features",
            "evaluate distribution shift by age and treatment modality",
            "avoid patient-facing recommendations or clinical utility claims",
        ],
        "not_requested": ["direct identifiers", "unbounded clinical note extraction"],
    },
]


def build_restricted_data_access_packet(
    *,
    output_path: str = DEFAULT_OUTPUT_PATH,
    md_path: str = DEFAULT_MD_PATH,
) -> dict[str, Any]:
    payload = {
        "schema_version": "restricted_data_access_packet_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready_for_future_access_request",
        "datasets": DATASET_PACKETS,
        "common_safeguards": [
            "use only de-identified or governed research data",
            "follow dataset-specific terms and data-use agreements",
            "store data outside git",
            "run leakage and target-compatibility checks before modeling",
            "publish only aggregate engineering metrics",
            "do not expose patient-level records in demos",
        ],
        "review_questions_for_future_advisor": [
            "Are the target labels clinically meaningful for monitor-only use?",
            "Which outputs should be hidden from patients and shown only to clinicians?",
            "Which thresholds require clinician/nurse review before demo use?",
            "Do failure cases suggest unsafe wording or inappropriate model scope?",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(_resolve(output_path), payload)
    _write_markdown(_resolve(md_path), payload)
    return payload


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Restricted Dataset Access Packet",
        "",
        CLAIM_BOUNDARY,
        "",
        "## Datasets",
        "",
    ]
    for dataset in payload["datasets"]:
        lines.extend([
            f"### {dataset['dataset']}",
            "",
            f"- Priority: `{dataset['priority']}`",
            f"- Why it matters: {dataset['why_it_matters']}",
            "- Fields requested: " + ", ".join(dataset["fields_requested"]),
            "- Not requested: " + ", ".join(dataset["not_requested"]),
            "",
            "Analysis plan:",
            "",
        ])
        lines.extend(f"- {item}" for item in dataset["analysis_plan"])
        lines.append("")
    lines.extend([
        "## Safeguards",
        "",
        *[f"- {item}" for item in payload["common_safeguards"]],
        "",
        "## Advisor Review Questions",
        "",
        *[f"- {item}" for item in payload["review_questions_for_future_advisor"]],
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate
