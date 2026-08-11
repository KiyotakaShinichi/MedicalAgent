"""Orchestrate repository-native NLCare engineering evaluations."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "Data/evals/governance/latest_nlcare_eval_run.json"
DEFAULT_MARKDOWN = ROOT / "reports/latest_nlcare_eval_run.md"
SEED = 20260811
CLAIM_BOUNDARY = (
    "Reproducible local engineering evaluation over synthetic/internal assets. A passing suite is not clinical validation, "
    "independent external evaluation, a production SLO, security certification, or healthcare deployment approval."
)


def run_evaluation(
    suites: set[str],
    *,
    output_path: str | Path = DEFAULT_JSON,
    markdown_path: str | Path = DEFAULT_MARKDOWN,
) -> dict[str, Any]:
    random.seed(SEED)
    selected = _expand_suites(suites)
    results: list[dict[str, Any]] = []
    for name in selected:
        results.append(_execute(name, _registry()[name]))
    failed = [row for row in results if row["status"] == "failed"]
    needs_attention = [row for row in results if row["reported_status"] in {"needs_attention", "failed"}]
    payload = {
        "schema_version": "nlcare_eval_run_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed" if failed else ("needs_attention" if needs_attention else "acceptable_internal_run"),
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "suite_version": "2026.08.next_generation.1",
        "random_seed": SEED,
        "selected_suites": selected,
        "provenance": _provenance(),
        "results": results,
        "summary": {
            "suite_count": len(results),
            "execution_failure_count": len(failed),
            "reported_needs_attention_count": len(needs_attention),
            "blocked_external": [
                "independently_authored external RAG holdout",
                "oncology clinician or nurse review",
                "genetic counselor VUS review",
                "managed-cloud deployment and traffic evidence",
            ],
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown = Path(markdown_path)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(_markdown(payload), encoding="utf-8")
    return payload


def _registry() -> dict[str, Callable[[], dict[str, Any]]]:
    from backend.services.adversarial_generalization_vnext import build_adversarial_generalization_vnext
    from backend.services.ai_trinity_tradeoff import write_ai_trinity_tradeoff
    from backend.services.automation_fault_injection_eval import build_automation_fault_injection_eval
    from backend.services.evaluation_dataset_integrity import write_integrity_report
    from backend.services.external_review_execution_readiness import build_readiness
    from backend.services.human_review_feedback_ingestion import build_human_review_feedback_ingestion
    from backend.services.rag_corpus_poisoning_eval import build_corpus_poisoning_eval
    from backend.services.rag_degradation_resilience_eval import build_rag_degradation_resilience_eval
    from backend.services.rag_failure_attribution_vnext import build_rag_failure_attribution
    from backend.services.saas_foundation_readiness import write_saas_foundation_readiness
    from backend.services.section_aware_retrieval_eval import run_section_aware_retrieval_eval
    from backend.services.synthetic_load_matrix import run_synthetic_load_matrix
    from backend.services.tenant_isolation_security_eval import build_tenant_isolation_security_eval

    return {
        "integrity": lambda: write_integrity_report(raise_on_failure=False),
        "ai_trinity": write_ai_trinity_tradeoff,
        "safety": build_adversarial_generalization_vnext,
        "security_tenant": build_tenant_isolation_security_eval,
        "security_poisoning": build_corpus_poisoning_eval,
        "rag_attribution": build_rag_failure_attribution,
        "rag_section_ablation": run_section_aware_retrieval_eval,
        "automation": build_automation_fault_injection_eval,
        "reliability_rag": build_rag_degradation_resilience_eval,
        "load": run_synthetic_load_matrix,
        "saas": write_saas_foundation_readiness,
        "external_review_readiness": build_readiness,
        "external_feedback": build_human_review_feedback_ingestion,
        "ml": _existing_ml_evidence,
        "xai": _existing_xai_evidence,
    }


def _expand_suites(suites: set[str]) -> list[str]:
    if not suites or "quick" in suites:
        return ["integrity", "ai_trinity", "security_tenant", "security_poisoning", "rag_attribution", "external_feedback"]
    if "full" in suites:
        return list(_registry())
    aliases = {
        "retrieval": ["rag_attribution", "rag_section_ablation"],
        "rag": ["ai_trinity", "rag_attribution", "rag_section_ablation", "reliability_rag"],
        "security": ["security_tenant", "security_poisoning"],
    }
    selected: list[str] = []
    for suite in sorted(suites):
        for name in aliases.get(suite, [suite]):
            if name not in _registry():
                raise ValueError(f"Unknown evaluation suite: {suite}")
            if name not in selected:
                selected.append(name)
    return selected


def _execute(name: str, function: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    started = perf_counter()
    try:
        result = function()
        if isinstance(result, Path):
            result = json.loads(result.read_text(encoding="utf-8"))
        return {
            "suite": name,
            "status": "completed",
            "reported_status": result.get("status"),
            "duration_ms": round((perf_counter() - started) * 1000, 3),
            "clinical_validation": result.get("clinical_validation", False),
            "headline": _headline(result),
        }
    except Exception as exc:
        return {
            "suite": name,
            "status": "failed",
            "reported_status": "failed",
            "duration_ms": round((perf_counter() - started) * 1000, 3),
            "clinical_validation": False,
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
        }


def _existing_ml_evidence() -> dict[str, Any]:
    return _validate_existing("Data/evals/models/latest_leakage_audit.json", "existing_ml_evidence_check")


def _existing_xai_evidence() -> dict[str, Any]:
    return _validate_existing("Data/evals/models/latest_xai_reliability_gate.json", "existing_xai_evidence_check")


def _validate_existing(relative: str, schema: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.exists():
        return {"status": "needs_attention", "clinical_validation": False, "missing": relative}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema_version": schema,
        "status": payload.get("status", "informational"),
        "clinical_validation": False,
        "source_artifact": relative,
        "source_generated_at": payload.get("generated_at"),
    }


def _headline(payload: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "status", "total_n", "pass_rate", "failure_count", "integrity_failure_count",
        "scenario_count", "passed_count", "control_count", "passed_control_count",
        "external_review_completed", "accepted_feedback_row_count",
    )
    return {key: payload.get(key) for key in keys if key in payload}


def _provenance() -> dict[str, Any]:
    from backend.services.agent_rag import _knowledge_snippets, knowledge_base_fingerprint
    from backend.services.rag_vector_index import rag_index_status

    corpus = _knowledge_snippets()
    return {
        "git_commit": _git_commit(),
        "working_tree_dirty": _git_dirty(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "environment_profile": os.getenv("NLCARE_ENV", "local_or_unspecified"),
        "knowledge_base_fingerprint": knowledge_base_fingerprint(),
        "rag_index": rag_index_status(corpus=corpus, knowledge_fingerprint=knowledge_base_fingerprint()),
        "dataset_registry": "config/evaluation_dataset_registry.json",
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def _git_dirty() -> bool | None:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except Exception:
        return None


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# NLCare Evaluation Run",
        "",
        f"- Status: `{payload['status']}`",
        f"- Generated: `{payload['generated_at']}`",
        f"- Commit: `{payload['provenance'].get('git_commit')}`",
        f"- KB fingerprint: `{payload['provenance'].get('knowledge_base_fingerprint')}`",
        f"- Clinical validation: `{str(payload['clinical_validation']).lower()}`",
        "",
        "## Suites",
        "",
        "| Suite | Execution | Reported status | Duration ms |",
        "|---|---:|---:|---:|",
    ]
    for row in payload["results"]:
        lines.append(f"| {row['suite']} | {row['status']} | {row['reported_status']} | {row['duration_ms']} |")
    lines.extend(["", "## Boundary", "", payload["claim_boundary"], ""])
    return "\n".join(lines)


__all__ = ["run_evaluation"]
