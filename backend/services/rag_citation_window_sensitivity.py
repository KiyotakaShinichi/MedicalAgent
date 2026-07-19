"""Citation-window sensitivity analysis for RAG baseline results.

This is an evaluation-only artifact.  It does not change live retrieval,
generation, source-tier filtering, or the patient agent.  It measures whether
smaller cited-context windows would reduce citation noise on the existing
internal goldset.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.rag_baseline_comparison import (
    COMPARISON_OUTPUT_PATH,
    _citation_precision,
    _expected_source_groups,
)


OUTPUT_PATH = Path("Data/evals/rag/latest_citation_window_sensitivity.json")
DOC_PATH = Path("docs/evals/citation_window_sensitivity.md")
FULL_STACK_CONFIG = "hybrid_rrf_query_rewrite_parent_child_source_tier"
WINDOWS = (1, 2, 3, 5)


def build_citation_window_sensitivity(
    *,
    input_path: str | Path = COMPARISON_OUTPUT_PATH,
    output_path: str | Path = OUTPUT_PATH,
    doc_path: str | Path = DOC_PATH,
) -> dict[str, Any]:
    payload = _read_json(Path(input_path))
    cases = list(payload.get("configurations", {}).get(FULL_STACK_CONFIG, {}).get("cases") or [])
    rows = [_score_window(k, cases) for k in WINDOWS]
    baseline = next((row for row in rows if row["cited_context_k"] == 5), None) or {}
    candidates = [
        row for row in rows
        if row["cited_context_k"] < 5
        and row["citation_precision"] >= baseline.get("citation_precision", 0)
    ]
    best = max(candidates or rows, key=lambda row: (row["citation_precision"], row["cited_window_support_rate"], -row["cited_context_k"]))
    promoted = (
        bool(candidates)
        and best["citation_precision_delta_vs_k5"] > 0
        and best["cited_window_support_delta_vs_k5"] >= -0.05
    )
    report = {
        "schema_version": "citation_window_sensitivity_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "acceptable" if rows else "needs_attention",
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "live_patient_route_changed": False,
        "retrieval_ranking_changed": False,
        "configuration": FULL_STACK_CONFIG,
        "case_count": len(cases),
        "baseline_cited_context_k": 5,
        "rows": rows,
        "recommended_cited_context_k": best["cited_context_k"] if rows else None,
        "promotion_recommendation": "candidate_for_live_ab_test" if promoted else "do_not_promote_without_more_evidence",
        "why": (
            "Citation precision can improve when fewer chunks are cited, but this does not prove retrieval improvement. "
            "Any live change should be A/B tested against generated-answer claim support and refusal behavior."
        ),
        "claim_boundary": (
            "This is internal engineering evidence only. It is not clinical validation, not a patient-safety claim, "
            "and not proof that the RAG stack is clinically grounded."
        ),
        "failure_examples": _failure_examples(best["cited_context_k"], cases) if rows else [],
    }
    _write_json(Path(output_path), report)
    _write_doc(Path(doc_path), report)
    return report


def _score_window(k: int, cases: list[dict[str, Any]]) -> dict[str, Any]:
    scored = []
    for case in cases:
        expected_groups = _expected_source_groups(case)
        retrieved = [{"source_id": source_id} for source_id in case.get("retrieved_source_ids") or []]
        expected_refusal = bool(case.get("expected_refusal_or_insufficient_evidence"))
        precision = _citation_precision(retrieved[:k], expected_groups, expected_refusal)
        support = precision > 0
        scored.append({
            "case_id": case.get("case_id"),
            "citation_precision": precision,
            "cited_window_supported": support,
        })
    citation_precision = _mean(row["citation_precision"] for row in scored)
    cited_window_support_rate = _mean(1.0 if row["cited_window_supported"] else 0.0 for row in scored)
    baseline_precision = _mean(
        _citation_precision(
            [{"source_id": source_id} for source_id in case.get("retrieved_source_ids") or []][:5],
            _expected_source_groups(case),
            bool(case.get("expected_refusal_or_insufficient_evidence")),
        )
        for case in cases
    )
    baseline_support = _mean(
        1.0 if _citation_precision(
            [{"source_id": source_id} for source_id in case.get("retrieved_source_ids") or []][:5],
            _expected_source_groups(case),
            bool(case.get("expected_refusal_or_insufficient_evidence")),
        ) > 0 else 0.0
        for case in cases
    )
    return {
        "cited_context_k": k,
        "citation_precision": round(citation_precision, 4),
        "citation_precision_delta_vs_k5": round(citation_precision - baseline_precision, 4),
        "cited_window_support_rate": round(cited_window_support_rate, 4),
        "cited_window_support_delta_vs_k5": round(cited_window_support_rate - baseline_support, 4),
        "low_precision_case_count": sum(1 for row in scored if row["citation_precision"] < 0.5),
    }


def _failure_examples(k: int, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples = []
    for case in cases:
        expected_groups = _expected_source_groups(case)
        retrieved = [{"source_id": source_id} for source_id in case.get("retrieved_source_ids") or []]
        precision = _citation_precision(
            retrieved[:k],
            expected_groups,
            bool(case.get("expected_refusal_or_insufficient_evidence")),
        )
        if precision < 0.5:
            examples.append({
                "case_id": case.get("case_id"),
                "category": case.get("category"),
                "expected_source_ids": case.get("expected_source_ids"),
                "retrieved_source_ids": case.get("retrieved_source_ids", [])[:k],
                "citation_precision_at_k": precision,
                "failure_reasons": case.get("failure_reasons"),
            })
    return examples[:12]


def _mean(values) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_doc(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Citation Window Sensitivity",
        "",
        report["claim_boundary"],
        "",
        f"- Status: `{report['status']}`",
        f"- Configuration: `{report['configuration']}`",
        f"- Case count: `{report['case_count']}`",
        f"- Recommended cited-context K: `{report['recommended_cited_context_k']}`",
        f"- Promotion recommendation: `{report['promotion_recommendation']}`",
        "",
        "| cited_context_k | citation_precision | delta_vs_k5 | cited_window_support_rate | low_precision_cases |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['cited_context_k']} | {row['citation_precision']} | "
            f"{row['citation_precision_delta_vs_k5']} | {row['cited_window_support_rate']} | "
            f"{row['low_precision_case_count']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        report["why"],
        "",
        "Do not present this as clinical validation or as proof that retrieval is solved.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


__all__ = ["build_citation_window_sensitivity"]
