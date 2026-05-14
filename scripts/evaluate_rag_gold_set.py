"""Evaluate the hand-labeled RAG gold set with the existing regression harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_GOLD_PATH = "evals/rag_gold_cases.json"
DEFAULT_OUTPUT_PATH = "Data/evals/rag/latest_rag_gold_eval.json"


def _to_regression_case(case: dict) -> dict:
    boundary = case.get("safety_boundary") or ""
    should_refuse = any(term in boundary for term in ["refuse", "block", "insufficient"])
    should_escalate = "escalate" in boundary
    requires_citations = int(case.get("minimum_citation_count") or 0) > 0
    expected_sources = case.get("expected_sources") or []
    answer_points = [] if should_refuse else (case.get("expected_answer_points") or [])
    return {
        "id": case.get("id"),
        "category": case.get("category"),
        "input": case.get("question"),
        "expected_behavior": f"Gold-set boundary: {boundary}",
        "expected_sources": expected_sources,
        "expected_context_keywords": answer_points[:5],
        "requires_citations": requires_citations,
        "required_evidence_type": "gold_set",
        "should_refuse": should_refuse,
        "should_escalate": should_escalate,
        "expected_refusal_type": _expected_refusal_type(boundary),
        "fallback_response": "I can help organize this for clinician review, but I cannot diagnose or decide treatment.",
    }


def _expected_refusal_type(boundary: str) -> str | None:
    if "security" in boundary or "privacy" in boundary or "block" in boundary:
        return "security_block"
    if "diagnosis" in boundary:
        return "diagnosis_refusal"
    if "medication" in boundary:
        return "medication_refusal"
    if "treatment" in boundary:
        return "treatment_refusal"
    if "insufficient" in boundary:
        return "insufficient_evidence"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the RAG gold-set evaluation.")
    parser.add_argument("--cases-path", default=DEFAULT_GOLD_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--live-agent", action="store_true")
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()

    from backend.services.rag_eval_suite import run_rag_eval_suite

    catalog = json.loads(Path(args.cases_path).read_text(encoding="utf-8"))
    cases = [_to_regression_case(case) for case in catalog.get("cases") or []]
    result = run_rag_eval_suite(output_path=args.output_path, csv_path=None, cases=cases, live_agent=args.live_agent)
    result["gold_set"] = {
        "path": args.cases_path,
        "schema_version": catalog.get("schema_version"),
        "case_count": len(cases),
        "claim_boundary": catalog.get("claim_boundary"),
    }
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_path).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    if args.print_summary:
        print(json.dumps({"summary": result.get("summary"), "gold_set": result.get("gold_set")}, indent=2))
    return 0 if (result.get("summary") or {}).get("status") != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
