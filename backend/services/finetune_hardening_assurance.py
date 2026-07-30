"""Evidence-based assurance summary for the behavior-only fine-tune scaffold."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.oncology_canonical_schema import ROOT_DIR


DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_finetune_hardening_assurance.json"
DEFAULT_DOC_PATH = "docs/finetune_hardening.md"

DATASET_CARD_PATH = "Data/finetune/prepared/dataset_card.json"
RUNTIME_PREFLIGHT_PATH = "Data/evals/models/latest_finetune_runtime_preflight.json"
PROMOTION_PATH = "Data/evals/models/latest_finetune_promotion_gate.json"
SEMANTIC_CONTAMINATION_PATH = (
    "Data/evals/models/latest_finetune_semantic_contamination.json"
)

CLAIM_BOUNDARY = (
    "This assurance artifact covers an internal, synthetic, behavior-only fine-tuning "
    "scaffold. It is not medical knowledge tuning, not clinical validation, not proof "
    "of real-world safety, and not permission for patient-facing use."
)


def build_finetune_hardening_assurance(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    doc_path: str | Path = DEFAULT_DOC_PATH,
) -> dict[str, Any]:
    dataset = _read(DATASET_CARD_PATH)
    runtime = _read(RUNTIME_PREFLIGHT_PATH)
    promotion = _read(PROMOTION_PATH)
    semantic_contamination = _read(SEMANTIC_CONTAMINATION_PATH)
    contamination = dataset.get("contamination_audit") or {}
    semantic_summary = semantic_contamination.get("summary") or {}
    semantic_screen = semantic_contamination.get("screen") or {}
    checks = [
        _check(
            "behavior_only_scope",
            dataset.get("dataset_purpose", "").lower().find("not medical knowledge") >= 0,
            DATASET_CARD_PATH,
            "Dataset remains limited to behavior and format tuning.",
        ),
        _check(
            "synthetic_data_disclosed",
            dataset.get("synthetic_or_source") == "all_synthetic",
            DATASET_CARD_PATH,
            "All tuning examples are declared synthetic.",
        ),
        _check(
            "exact_holdout_contamination_clear",
            int(contamination.get("exact_overlap_count") or 0) == 0,
            DATASET_CARD_PATH,
            "No exact normalized overlap with registered RAG/safety holdouts.",
        ),
        _check(
            "semantic_similarity_screen_completed",
            semantic_screen.get("semantic_similarity_proxy_completed") is True,
            SEMANTIC_CONTAMINATION_PATH,
            "Word/character TF-IDF screening must cover train and registered evaluation text.",
        ),
        _check(
            "semantic_flags_cleared_for_candidate",
            semantic_summary.get("adjudication_cleared_for_candidate") is True,
            SEMANTIC_CONTAMINATION_PATH,
            "Every near-match requires review; contaminated or ambiguous pairs require remediation.",
        ),
        _check(
            "base_and_tokenizer_revisions_pinned",
            bool((runtime.get("candidate") or {}).get("revision"))
            and bool((runtime.get("candidate") or {}).get("tokenizer_revision")),
            RUNTIME_PREFLIGHT_PATH,
            "Immutable model and tokenizer revisions are recorded.",
        ),
        _check(
            "isolated_training_runtime_ready",
            runtime.get("ready_for_offline_experiment") is True,
            RUNTIME_PREFLIGHT_PATH,
            "PEFT dependencies and runtime probe must pass before training.",
        ),
        _check(
            "baseline_candidate_generations_complete",
            bool((promotion.get("evidence") or {}).get("baseline_present"))
            and bool((promotion.get("evidence") or {}).get("candidate_present")),
            PROMOTION_PATH,
            "Matched baseline and candidate generations are required.",
        ),
        _check(
            "candidate_generation_lineage_verified",
            bool((promotion.get("evidence") or {}).get("candidate_generation_lineage_verified")),
            PROMOTION_PATH,
            "Generation manifest must bind model, revisions, holdout hash, and generation hash.",
        ),
        _check(
            "candidate_memorization_audit_complete",
            bool((promotion.get("evidence") or {}).get("candidate_memorization_audit_completed")),
            PROMOTION_PATH,
            "Exact train-output memorization must be checked before shadow promotion.",
        ),
        _check(
            "paired_statistical_lift_proven",
            promotion.get("behavior_improvement_statistically_proven") is True,
            PROMOTION_PATH,
            "Exact paired McNemar/binomial evidence is required in addition to raw lift.",
        ),
        _check(
            "patient_facing_promotion_blocked",
            promotion.get("patient_facing_promotion_allowed") is False,
            PROMOTION_PATH,
            "Any promotion remains offline/shadow only.",
        ),
    ]
    passed = sum(1 for item in checks if item["passed"])
    blocking_ids = [
        item["id"]
        for item in checks
        if not item["passed"]
    ]
    payload = {
        "schema_version": "finetune_hardening_assurance_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong" if not blocking_ids else "needs_attention",
        "clinical_validation": False,
        "model_trained": bool(runtime.get("model_trained")),
        "adapter_created": bool(runtime.get("adapter_created")),
        "promotion_decision": promotion.get("decision") or "HOLD",
        "patient_facing_promotion_allowed": False,
        "summary": {
            "check_count": len(checks),
            "passed_count": passed,
            "pass_rate": round(passed / len(checks), 4),
            "blocking_gap_count": len(blocking_ids),
            "blocking_gap_ids": blocking_ids,
            "semantic_similarity_screen_completed": (
                semantic_screen.get("semantic_similarity_proxy_completed") is True
            ),
            "semantic_flagged_pair_count": int(
                semantic_summary.get("flagged_pair_count") or 0
            ),
            "semantic_unresolved_pair_count": int(
                semantic_summary.get("unresolved_pair_count") or 0
            ),
            "semantic_review_completed": (
                semantic_summary.get("review_completed") is True
            ),
            "semantic_adjudication_cleared_for_candidate": (
                semantic_summary.get("adjudication_cleared_for_candidate") is True
            ),
            "semantic_artifact_flag_rows_capped": (
                semantic_summary.get("artifact_flag_rows_capped") is True
            ),
            "semantic_contamination_absence_proven": False,
        },
        "checks": checks,
        "promotion_contract": {
            "minimum_paired_cases": 50,
            "minimum_cases_per_behavior": 5,
            "minimum_behavior_lift": 0.02,
            "maximum_paired_p_value": 0.05,
            "maximum_output_p95_ratio": 1.5,
            "unsafe_leakage_allowed": 0.0,
            "claim_boundary_compliance_required": 1.0,
            "generation_manifest_required": True,
            "train_output_memorization_audit_required": True,
            "semantic_similarity_screen_required": True,
            "semantic_flag_adjudication_required": True,
            "promotion_scope": "offline_shadow_only",
        },
        "next_actions": [
            "Provision an isolated pinned PEFT runtime and make the runtime preflight pass.",
            "Generate matched baseline and candidate outputs using the frozen internal holdout.",
            "File generation manifests with model, revision, adapter, generation, and holdout hashes.",
            "Run the paired promotion gate and inspect every safety and behavior regression.",
            "Adjudicate every semantic-similarity flag; TF-IDF screening alone is insufficient.",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write(output_path, payload)
    _write_doc(doc_path, payload)
    return payload


def _check(identifier: str, passed: bool, evidence: str, detail: str) -> dict[str, Any]:
    return {
        "id": identifier,
        "passed": bool(passed),
        "evidence_artifact": evidence,
        "detail": detail,
    }


def _read(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    full = candidate if candidate.is_absolute() else ROOT_DIR / candidate
    if not full.exists():
        return {}
    try:
        parsed = json.loads(full.read_text(encoding="utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _write(path: str | Path, payload: dict[str, Any]) -> None:
    candidate = Path(path)
    full = candidate if candidate.is_absolute() else ROOT_DIR / candidate
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_doc(path: str | Path, payload: dict[str, Any]) -> None:
    candidate = Path(path)
    full = candidate if candidate.is_absolute() else ROOT_DIR / candidate
    full.parent.mkdir(parents=True, exist_ok=True)
    failed = [item for item in payload["checks"] if not item["passed"]]
    lines = [
        "# Fine-Tuning Hardening",
        "",
        "This is an internal, synthetic, behavior-only scaffold. It does not tune "
        "medical authority and cannot be promoted to patient-facing use.",
        "",
        f"- Status: `{payload['status']}`",
        f"- Promotion decision: `{payload['promotion_decision']}`",
        f"- Checks passed: `{payload['summary']['passed_count']}/{payload['summary']['check_count']}`",
        "",
        "## Open Gates",
        "",
    ]
    lines.extend(
        f"- `{item['id']}`: {item['detail']} Evidence: `{item['evidence_artifact']}`"
        for item in failed
    )
    lines.extend(
        [
            "",
            "## Promotion Meaning",
            "",
            "A `PROMOTE` result means eligible for an offline shadow experiment only. "
            "It does not mean clinically validated, safe for patients, or approved for deployment.",
            "",
        ]
    )
    full.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "DEFAULT_DOC_PATH",
    "DEFAULT_OUTPUT_PATH",
    "build_finetune_hardening_assurance",
]
