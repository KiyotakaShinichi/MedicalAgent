"""Run the held-out RAG baseline comparison.

If ``Data/evals/rag/retrieval_goldset_holdout_v2.jsonl`` does NOT
exist, this script emits a **readiness artifact** that explicitly
reports ``completed: false``.  It does NOT manufacture an external
result by reusing the internal goldset.

If the holdout file exists, the script runs the same baseline
comparison logic as ``run_rag_baseline_comparison.py`` against the
holdout file and writes:

  Data/evals/rag/latest_rag_holdout_baseline_comparison.json
  Data/evals/rag/latest_rag_holdout_baseline_failures.json

Even in the completed case, the artifact carries
``external_author_eval_completed: true`` only when EVERY case in the
holdout file is ``internal_vs_external_authored: "external"`` and
``was_used_for_tuning: false`` AND none of the case ``query`` fields
contain the substring ``<PLACEHOLDER>``.  Any violation downgrades
the artifact to ``completed: false`` with a documented reason.

See ``docs/evals/no_read_rag_goldset_protocol.md`` for the author
protocol this runner enforces.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.rag_baseline_comparison import (  # noqa: E402
    run_rag_baseline_comparison,
)


REQUIRED_HOLDOUT_PATH = Path("Data/evals/rag/retrieval_goldset_holdout_v2.jsonl")
TEMPLATE_PATH = Path("Data/evals/rag/retrieval_goldset_holdout_v2_template.jsonl")
PROTOCOL_DOC = Path("docs/evals/no_read_rag_goldset_protocol.md")
COMPARISON_OUTPUT_PATH = Path("Data/evals/rag/latest_rag_holdout_baseline_comparison.json")
FAILURES_OUTPUT_PATH = Path("Data/evals/rag/latest_rag_holdout_baseline_failures.json")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _placeholder_violations(rows: list[dict[str, Any]]) -> list[str]:
    """Return case_ids whose query / source_ids still contain a placeholder marker."""
    bad: list[str] = []
    for row in rows:
        text = json.dumps(row, ensure_ascii=False)
        if "<PLACEHOLDER" in text or "PLACEHOLDER:" in text or "<reviewer_role_descriptor>" in text:
            bad.append(str(row.get("case_id") or "?"))
    return bad


def _non_external_cases(rows: list[dict[str, Any]]) -> list[str]:
    return [
        str(row.get("case_id") or "?")
        for row in rows
        if str(row.get("internal_vs_external_authored", "")).lower() != "external"
    ]


def _tuning_tainted_cases(rows: list[dict[str, Any]]) -> list[str]:
    return [
        str(row.get("case_id") or "?")
        for row in rows
        if bool(row.get("was_used_for_tuning", False))
    ]


def _readiness_artifact(reason: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "rag_holdout_baseline_comparison_v1",
        "status": "ready_for_external_authoring",
        "completed": False,
        "external_author_eval_completed": False,
        "clinical_validation": False,
        "reason": reason,
        "required_path": str(REQUIRED_HOLDOUT_PATH).replace("\\", "/"),
        "protocol_doc": str(PROTOCOL_DOC).replace("\\", "/"),
        "template_path": str(TEMPLATE_PATH).replace("\\", "/"),
        "claim_boundary": (
            "Held-out RAG comparison is engineering infrastructure only. "
            "Even a completed run does NOT establish clinical validity, "
            "real-world safety, or production healthcare readiness. The "
            "readiness state reports preparation, not validation."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    return payload


def _completed_artifact(comparison: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap the in-sample comparison shape with completed-holdout metadata."""
    summary = comparison.get("summary") or {}
    payload = {
        "schema_version": "rag_holdout_baseline_comparison_v1",
        "status": comparison.get("status") or "acceptable",
        "completed": True,
        "external_author_eval_completed": True,
        "clinical_validation": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "goldset_path": str(REQUIRED_HOLDOUT_PATH).replace("\\", "/"),
        "protocol_doc": str(PROTOCOL_DOC).replace("\\", "/"),
        "total_n": len(cases),
        "authored_by_summary": _authored_by_counts(cases),
        "internal_vs_external_authored": "external",
        "was_used_for_tuning": False,
        "category_counts": _category_counts(cases),
        "contamination_note": (
            "Held-out external-authored cases following docs/evals/no_read_rag_goldset_protocol.md. "
            "These cases were NOT used to tune retrieval weights, alias map, prompts, or thresholds. "
            "They have not been read by anyone who has tuned the system."
        ),
        "claim_boundary": (
            "Engineering evaluation only.  A high held-out recall does NOT establish "
            "clinical validity, real-world safety, or production healthcare readiness."
        ),
        "recall_at_5": summary.get("full_stack_recall_at_5") or summary.get("full_stack_recall_at_10"),
        "recall_at_10": summary.get("full_stack_recall_at_10"),
        "best_recall_at_10": summary.get("best_recall_at_10"),
        "bm25_recall_at_10": summary.get("bm25_recall_at_10"),
        "full_stack_recall_at_10": summary.get("full_stack_recall_at_10"),
        "full_stack_mrr": summary.get("full_stack_mrr"),
        "full_stack_ndcg_at_10": summary.get("full_stack_ndcg_at_10"),
        "citation_precision": summary.get("citation_precision"),
        "claim_support_rate": summary.get("claim_support_rate"),
        "unsupported_context_rate": summary.get("unsupported_context_rate"),
        "refusal_correctness": summary.get("refusal_correctness"),
        "source_tier_correctness": summary.get("source_tier_correctness"),
        "latency_p50_ms": summary.get("latency_p50_ms"),
        "latency_p95_ms": summary.get("latency_p95_ms"),
        "improvement_proven_vs_bm25_holdout": bool(summary.get("improvement_proven_vs_bm25")),
        "configurations": comparison.get("configurations") or {},
        "rows": comparison.get("rows") or [],
        "underlying_in_sample_summary": summary,
    }
    return payload


def _authored_by_counts(cases: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in cases:
        key = str(c.get("authored_by") or "unknown")
        out[key] = out.get(key, 0) + 1
    return out


def _category_counts(cases: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in cases:
        key = str(c.get("category") or "uncategorised")
        out[key] = out.get(key, 0) + 1
    return out


def run(
    *,
    holdout_path: Path = REQUIRED_HOLDOUT_PATH,
    comparison_output: Path = COMPARISON_OUTPUT_PATH,
    failures_output: Path = FAILURES_OUTPUT_PATH,
) -> dict[str, Any]:
    comparison_output.parent.mkdir(parents=True, exist_ok=True)

    if not holdout_path.exists():
        report = _readiness_artifact(
            reason="holdout file not found; complete the no-read protocol first",
        )
        comparison_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        failures_output.write_text(
            json.dumps({"completed": False, "reason": "holdout file not found", "failures": []}, indent=2),
            encoding="utf-8",
        )
        return report

    cases = _load_jsonl(holdout_path)
    if not cases:
        report = _readiness_artifact(
            reason="holdout file exists but has zero cases",
            extra={"n_cases": 0},
        )
        comparison_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        failures_output.write_text(
            json.dumps({"completed": False, "reason": "empty file", "failures": []}, indent=2),
            encoding="utf-8",
        )
        return report

    placeholder_bad = _placeholder_violations(cases)
    non_external = _non_external_cases(cases)
    tuning_tainted = _tuning_tainted_cases(cases)

    if placeholder_bad or non_external or tuning_tainted:
        reason_parts = []
        if placeholder_bad:
            reason_parts.append(f"{len(placeholder_bad)} cases still contain placeholders")
        if non_external:
            reason_parts.append(f"{len(non_external)} cases not marked external")
        if tuning_tainted:
            reason_parts.append(f"{len(tuning_tainted)} cases marked was_used_for_tuning=true")
        report = _readiness_artifact(
            reason="; ".join(reason_parts),
            extra={
                "n_cases": len(cases),
                "cases_with_placeholders": placeholder_bad,
                "cases_not_external": non_external,
                "cases_used_for_tuning": tuning_tainted,
            },
        )
        comparison_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        failures_output.write_text(
            json.dumps(
                {
                    "completed": False,
                    "reason": report["reason"],
                    "n_cases": len(cases),
                    "failures": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return report

    # All gating passes — run the real comparison on the holdout file.
    comparison = run_rag_baseline_comparison(
        goldset_path=holdout_path,
        comparison_output_path=comparison_output,
        failures_output_path=failures_output,
    )
    completed = _completed_artifact(comparison, cases)
    comparison_output.write_text(json.dumps(completed, indent=2), encoding="utf-8")
    # failures_output was already written by run_rag_baseline_comparison
    return completed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", default=str(REQUIRED_HOLDOUT_PATH))
    parser.add_argument("--out", default=str(COMPARISON_OUTPUT_PATH))
    parser.add_argument("--failures", default=str(FAILURES_OUTPUT_PATH))
    args = parser.parse_args()

    result = run(
        holdout_path=Path(args.holdout),
        comparison_output=Path(args.out),
        failures_output=Path(args.failures),
    )
    if result.get("completed"):
        print(f"completed: yes  n={result.get('total_n')}  "
              f"BM25={result.get('bm25_recall_at_10')}  "
              f"full={result.get('full_stack_recall_at_10')}  "
              f"improvement_proven={result.get('improvement_proven_vs_bm25_holdout')}")
    else:
        print(f"completed: no  reason={result.get('reason')}")
        print(f"  required_path: {result.get('required_path')}")
        print(f"  protocol_doc:  {result.get('protocol_doc')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
