from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.artifact_manifest import build_artifact_manifest, freshness_status


DEFAULT_JSON_PATH = "Data/evals/benchmark/latest_benchmark_summary.json"
DEFAULT_MD_PATH = "benchmarks/benchmark_report.md"
DEFAULT_CSV_PATH = "benchmarks/benchmark_results.csv"

ROOT_DIR = Path(__file__).resolve().parents[2]


BENCHMARK_SPECS: list[dict[str, Any]] = [
    {
        "id": "safety_red_team",
        "title": "Safety red-team",
        "path": "Data/evals/safety/latest_safety_benchmark.json",
        "fallback": "Data/evals/safety/latest_safety_red_team.json",
        "tier": "critical",
        "metrics": {
            "pass_rate": ["summary", "pass_rate"],
            "failed_cases": ["summary", "failed_cases"],
            "total_cases": ["summary", "total_cases"],
        },
    },
    {
        "id": "adversarial",
        "title": "Adversarial prompt/jailbreak",
        "path": "Data/evals/safety/latest_adversarial_eval.json",
        "tier": "critical",
        "metrics": {
            "attack_block_rate": ["summary", "pass_rate"],
            "failed_cases": ["summary", "failed_cases"],
        },
    },
    {
        "id": "multilingual_refusal",
        "title": "Multilingual refusal routing",
        "path": "Data/evals/safety/latest_multilingual_refusal_eval.json",
        "tier": "critical",
        "metrics": {
            "pass_rate": ["summary", "pass_rate"],
            "passed": ["summary", "passed"],
            "case_count": ["summary", "case_count"],
        },
    },
    {
        "id": "rag_regression",
        "title": "RAG regression",
        "path": "Data/evals/rag/latest_rag_benchmark.json",
        "fallback": "Data/evals/rag/latest_rag_eval.json",
        "tier": "critical",
        "metrics": {
            "pass_rate": ["summary", "pass_rate"],
            "citation_coverage_rate": ["summary", "citation_coverage_rate"],
            "expected_source_hit_rate": ["summary", "expected_source_hit_rate"],
            "unsafe_answer_rate": ["summary", "unsafe_answer_rate"],
            "average_grounding_score": ["summary", "average_grounding_score"],
        },
    },
    {
        "id": "rag_gold",
        "title": "Hand-labeled RAG gold set",
        "path": "Data/evals/rag/latest_rag_gold_eval.json",
        "tier": "critical",
        "metrics": {
            "pass_rate": ["summary", "pass_rate"],
            "expected_source_hit_rate": ["summary", "expected_source_hit_rate"],
            "case_count": ["gold_set", "case_count"],
            "unsafe_answer_rate": ["summary", "unsafe_answer_rate"],
        },
    },
    {
        "id": "tool_action_benchmark",
        "title": "Patient-support tool action benchmark",
        "path": "Data/evals/tool_actions/latest_tool_action_benchmark.json",
        "tier": "critical",
        "metrics": {
            "pass_rate": ["summary", "pass_rate"],
            "case_count": ["summary", "case_count"],
            "average_latency_ms": ["summary", "average_latency_ms"],
            "max_latency_ms": ["summary", "max_latency_ms"],
        },
    },
    {
        "id": "mle_readiness",
        "title": "MLE readiness gate",
        "path": "Data/mle_monitoring/latest_mle_readiness.json",
        "tier": "critical",
        "metrics": {
            "hard_gate_status": ["hard_gate_status"],
            "release_recommendation": ["release_recommendation"],
            "safety_regression": ["category_statuses", "safety_regression"],
            "monitoring": ["category_statuses", "monitoring"],
        },
    },
    {
        "id": "mle_readiness_realism_candidate",
        "title": "MLE readiness - realism candidate",
        "path": "Data/mle_monitoring/latest_mle_readiness_realism_candidate.json",
        "tier": "supporting",
        "metrics": {
            "hard_gate_status": ["hard_gate_status"],
            "release_recommendation": ["release_recommendation"],
            "safety_regression": ["category_statuses", "safety_regression"],
            "realism": ["category_statuses", "realism"],
            "monitoring": ["category_statuses", "monitoring"],
        },
    },
    {
        "id": "model_benchmark",
        "title": "Model benchmark",
        "path": "Data/evals/models/latest_model_benchmark.json",
        "tier": "critical",
        "metrics": {
            "synthetic_champion_auroc": ["synthetic_classification", 0, "roc_auc"],
            "synthetic_champion_auprc": ["synthetic_classification", 0, "auprc"],
            "synthetic_champion_brier": ["synthetic_classification", 0, "brier"],
            "external_breastdcedl_auroc": ["external_baselines", 0, "roc_auc"],
        },
    },
    {
        "id": "current_vs_realism_candidate",
        "title": "Current vs realism-calibrated candidate",
        "path": "Data/mle_monitoring/current_vs_realism_candidate.json",
        "tier": "critical",
        "metrics": {
            "decision": ["recommendation", "decision"],
            "auc_delta": ["recommendation", "auc_delta"],
            "realism_delta": ["recommendation", "realism_delta"],
            "candidate_alignment": ["candidate", "realism_alignment_score"],
            "current_alignment": ["current", "realism_alignment_score"],
        },
    },
    {
        "id": "synthetic_realism_candidate",
        "title": "Synthetic realism candidate",
        "path": "Data/mle_monitoring/synthetic_realism_candidate_report.json",
        "tier": "critical",
        "metrics": {
            "alignment_score": ["realism_alignment_score"],
            "training_patients": ["training_patients"],
            "threshold_coverage_status": ["threshold_coverage", "status"],
        },
    },
    {
        "id": "noise_eval",
        "title": "Noise robustness",
        "path": "Data/mle_monitoring/noise_eval_report.json",
        "tier": "supporting",
        "metrics": {
            "max_auroc_drop": ["max_auroc_drop"],
            "test_patients": ["test_patients"],
            "test_rows": ["test_rows"],
        },
    },
    {
        "id": "temporal_eval",
        "title": "Temporal generalization",
        "path": "Data/mle_monitoring/temporal_eval_report.json",
        "tier": "supporting",
        "metrics": {
            "temporal_auroc": ["temporal_split", "eval_auroc"],
            "random_baseline_auroc": ["random_split_baseline", "eval_auroc"],
            "generalization_gap": ["generalization_gap", "temporal_minus_random_auroc"],
        },
    },
    {
        "id": "calibration_eval",
        "title": "Calibration reliability",
        "path": "Data/mle_monitoring/calibration_eval_report.json",
        "tier": "supporting",
        "metrics": {
            "best_method": ["best_method"],
            "best_ece": ["best_ece"],
            "best_brier": ["best_brier"],
        },
    },
    {
        "id": "clinician_summary",
        "title": "Clinician summary quality",
        "path": "Data/evals/clinician_summary/latest_clinician_summary_eval.json",
        "tier": "supporting",
        "metrics": {
            "decision_accuracy": ["decision_accuracy"],
            "summary_completeness_rate_legitimate": ["summary_completeness_rate_legitimate"],
            "unsafe_leakage_rate": ["unsafe_leakage_rate"],
            "unsafe_detection_recall": ["unsafe_detection_recall"],
        },
    },
    {
        "id": "llm_judge",
        "title": "Optional LLM judge",
        "path": "Data/evals/llm_judge/latest_llm_judge_eval.json",
        "tier": "optional",
        "metrics": {
            "coverage_rate": ["summary", "coverage_rate"],
            "pass_rate": ["summary", "pass_rate"],
            "unsafe_medical_advice_rate": ["summary", "unsafe_medical_advice_rate"],
        },
    },
    {
        "id": "public_imaging_manifest",
        "title": "Public imaging readiness",
        "path": "Data/public_imaging/public_imaging_manifest.json",
        "tier": "supporting",
        "metrics": {
            "available_dataset_count": ["available_dataset_count"],
            "recommended_next_task": ["recommended_next_task"],
        },
    },
    {
        "id": "ultrasound_baseline",
        "title": "Ultrasound baseline",
        "path": "Data/public_imaging/ultrasound_baseline/metrics.json",
        "tier": "optional",
        "metrics": {
            "reason": ["reason"],
            "macro_f1": ["macro_f1"],
            "balanced_accuracy": ["balanced_accuracy"],
        },
    },
    {
        "id": "ct_lesion_workflow",
        "title": "CT lesion workflow",
        "path": "Data/public_imaging/ct_lesion_workflow/report.json",
        "tier": "optional",
        "metrics": {
            "reason": ["reason"],
            "study_count": ["study_count"],
            "workflow_stage": ["workflow_stage"],
        },
    },
]


def build_benchmark_registry(
    *,
    output_path: str = DEFAULT_JSON_PATH,
    report_path: str = DEFAULT_MD_PATH,
    csv_path: str = DEFAULT_CSV_PATH,
    freshness_ttl_seconds: int = 24 * 60 * 60,
) -> dict[str, Any]:
    rows = [_build_row(spec, freshness_ttl_seconds) for spec in BENCHMARK_SPECS]
    issues = _collect_issues(rows)
    payload: dict[str, Any] = {
        **build_artifact_manifest(
            dataset_paths={
                "benchmark_sources": "benchmarks",
                "rag_gold_cases": "evals/rag_gold_cases.json",
                "safety_cases": "evals/safety_red_team_cases.json",
                "synthetic_realism_candidate": "Data/complete_synthetic_breast_journeys_realism_v2/temporal_ml_rows.csv",
            },
            ttl_seconds=freshness_ttl_seconds,
        ),
        "schema_version": "benchmark_registry_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": _overall_status(rows),
        "critical_status": _tier_status(rows, "critical"),
        "supporting_status": _tier_status(rows, "supporting"),
        "optional_status": _tier_status(rows, "optional"),
        "benchmarks": rows,
        "issues": issues,
        "next_actions": _next_actions(rows, issues),
        "claim_boundary": (
            "Benchmarks are engineering evidence only. They test reproducibility, "
            "guardrails, retrieval behavior, calibration, and synthetic realism; "
            "they do not establish clinical safety or clinical validity."
        ),
        "report_path": report_path,
        "csv_path": csv_path,
    }
    _write_json(output_path, payload)
    _write_csv(csv_path, rows)
    _write_markdown(report_path, payload)
    return payload


def _build_row(spec: dict[str, Any], ttl_seconds: int) -> dict[str, Any]:
    payload, source_path, load_status = _load_with_fallback(spec["path"], spec.get("fallback"))
    generated_at = payload.get("generated_at") or _dig(payload, ["artifact_freshness", "generated_at"])
    artifact_ttl = _dig(payload, ["artifact_freshness", "ttl_seconds"]) or ttl_seconds
    freshness = freshness_status(generated_at, int(artifact_ttl)) if payload else "unknown"
    raw_status = _extract_status(payload, load_status)
    normalized_status = _normalize_status(raw_status, freshness, spec["tier"])
    metrics = {name: _dig(payload, path) for name, path in spec.get("metrics", {}).items()}
    return {
        "id": spec["id"],
        "title": spec["title"],
        "tier": spec["tier"],
        "status": normalized_status,
        "source_status": raw_status,
        "freshness": freshness,
        "generated_at": generated_at,
        "source_path": source_path,
        "metrics": metrics,
        "limitations": payload.get("limitations") or [],
        "claim_boundary": payload.get("claim_boundary"),
    }


def _load_with_fallback(path: str, fallback: str | None = None) -> tuple[dict[str, Any], str, str]:
    payload, status = _load_json(path)
    if status == "missing" and fallback:
        fallback_payload, fallback_status = _load_json(fallback)
        if fallback_status != "missing":
            return fallback_payload, fallback, fallback_status
    return payload, path, status


def _load_json(path: str) -> tuple[dict[str, Any], str]:
    file_path = ROOT_DIR / path
    if not file_path.exists():
        return {"status": "missing", "path": path}, "missing"
    try:
        return json.loads(file_path.read_text(encoding="utf-8")), "loaded"
    except json.JSONDecodeError as exc:
        return {"status": "error", "path": path, "error": str(exc)}, "error"


def _extract_status(payload: dict[str, Any], load_status: str) -> str:
    if load_status != "loaded":
        return load_status
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    return str(summary.get("status") or payload.get("status") or "available")


def _normalize_status(raw_status: str, freshness: str, tier: str) -> str:
    raw = (raw_status or "unknown").lower()
    if raw in {"missing", "error", "failed"}:
        return raw
    if raw == "unavailable":
        return "optional_unavailable" if tier == "optional" else "unavailable"
    if freshness == "stale":
        return "stale"
    return raw


def _overall_status(rows: list[dict[str, Any]]) -> str:
    critical = [row for row in rows if row["tier"] == "critical"]
    critical_status = _tier_status(rows, "critical")
    if critical_status in {"failed", "missing", "error", "unavailable"}:
        return "blocked"
    if any(row["status"] in {"needs_attention", "unideal", "stale"} for row in critical):
        return "needs_attention"
    if any(row["status"] in {"acceptable", "available"} for row in critical):
        return "acceptable"
    return "strong"


def _tier_status(rows: list[dict[str, Any]], tier: str) -> str:
    statuses = {row["status"] for row in rows if row["tier"] == tier}
    if not statuses:
        return "not_configured"
    for status in ("error", "missing", "failed", "unavailable"):
        if status in statuses:
            return status
    if "stale" in statuses:
        return "stale"
    if statuses & {"needs_attention", "unideal"}:
        return "needs_attention"
    if statuses <= {"strong", "passed", "robust", "stable"}:
        return "strong"
    if statuses <= {"strong", "passed", "robust", "stable", "acceptable", "available", "optional_unavailable"}:
        return "acceptable"
    return "needs_attention"


def _collect_issues(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    bad_statuses = {"error", "missing", "failed", "unavailable", "needs_attention", "unideal", "stale"}
    for row in rows:
        if row["status"] in bad_statuses:
            severity = "high" if row["tier"] == "critical" and row["status"] not in {"stale"} else "medium"
            issues.append({
                "benchmark_id": row["id"],
                "severity": severity,
                "status": row["status"],
                "message": _issue_message(row),
            })
    return issues


def _issue_message(row: dict[str, Any]) -> str:
    if row["status"] == "stale":
        return "Artifact is older than the freshness TTL; rerun this benchmark before quoting it."
    if row["status"] in {"missing", "unavailable"}:
        return "Artifact is not available; dashboard should show this as missing, not hidden validation."
    if row["id"] == "mle_readiness":
        return "MLE readiness has one or more unideal categories; inspect category_statuses before promotion."
    return "Benchmark needs review before using it as supporting evidence."


def _next_actions(rows: list[dict[str, Any]], issues: list[dict[str, Any]]) -> list[str]:
    by_id = {row["id"]: row for row in rows}
    actions: list[str] = []
    candidate = by_id.get("current_vs_realism_candidate", {})
    if _dig(candidate, ["metrics", "decision"]) == "promote_candidate_after_review":
        actions.append(
            "Promote the realism-calibrated synthetic candidate after reviewing threshold coverage and model-card language."
        )
    if by_id.get("public_imaging_manifest", {}).get("metrics", {}).get("available_dataset_count") == 0:
        actions.append(
            "Download one public imaging dataset into Datasets/ first; BUSI is the lightest hardware-friendly ultrasound start."
        )
    if by_id.get("llm_judge", {}).get("status") in {"optional_unavailable", "unavailable", "missing"}:
        actions.append(
            "Keep LLM-judge optional, or configure a provider and rerun it as a heuristic grounding review."
        )
    if any(issue["benchmark_id"] == "mle_readiness" for issue in issues):
        actions.append(
            "Rerun MLE readiness after benchmark refresh; it should consume the latest realism, noise, temporal, and safety artifacts."
        )
    if not actions:
        actions.append("No hard benchmark blocker detected; focus next on public-data-backed validation and UI polish.")
    return actions


def _dig(payload: Any, path: list[Any]) -> Any:
    value = payload
    for key in path:
        if isinstance(value, dict):
            value = value.get(key)
        elif isinstance(value, list) and isinstance(key, int) and 0 <= key < len(value):
            value = value[key]
        else:
            return None
    return value


def _write_json(path: str, payload: dict[str, Any]) -> None:
    output = ROOT_DIR / path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    output = ROOT_DIR / path
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "benchmark_id",
                "title",
                "tier",
                "status",
                "source_status",
                "freshness",
                "source_path",
                "metric",
                "value",
            ],
        )
        writer.writeheader()
        for row in rows:
            metrics = row.get("metrics") or {"status": row.get("status")}
            for metric, value in metrics.items():
                writer.writerow({
                    "benchmark_id": row["id"],
                    "title": row["title"],
                    "tier": row["tier"],
                    "status": row["status"],
                    "source_status": row["source_status"],
                    "freshness": row["freshness"],
                    "source_path": row["source_path"],
                    "metric": metric,
                    "value": value,
                })


def _write_markdown(path: str, payload: dict[str, Any]) -> None:
    output = ROOT_DIR / path
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MedicalAgent Benchmark Registry",
        "",
        f"Generated at: {payload['generated_at']}",
        "",
        f"Overall status: **{payload['status']}**",
        f"Critical status: **{payload['critical_status']}**",
        "",
        payload["claim_boundary"],
        "",
        "## Benchmark Matrix",
        "",
        "| Benchmark | Tier | Status | Freshness | Key metrics | Source |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in payload["benchmarks"]:
        metric_text = "; ".join(
            f"{key}={_format_metric(value)}"
            for key, value in (row.get("metrics") or {}).items()
            if value is not None
        ) or "no extracted metrics"
        lines.append(
            f"| {row['title']} | {row['tier']} | {row['status']} | "
            f"{row['freshness']} | {metric_text} | `{row['source_path']}` |"
        )
    lines.extend(["", "## Issues"])
    if payload["issues"]:
        for issue in payload["issues"]:
            lines.append(
                f"- {issue['severity']}: {issue['benchmark_id']} "
                f"({issue['status']}) - {issue['message']}"
            )
    else:
        lines.append("- No current hard issues detected.")
    lines.extend(["", "## Next Actions"])
    for action in payload["next_actions"]:
        lines.append(f"- {action}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
