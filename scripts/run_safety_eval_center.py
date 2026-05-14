"""Run the safety, RAG, and drift evaluation suites and write all artifacts.

This is the headless CLI entry point used by CI and demo seeding to regenerate
every artifact consumed by the Safety & Evaluation Center page.

Usage::

    python -m scripts.run_safety_eval_center            # all suites
    python -m scripts.run_safety_eval_center --only safety
    python -m scripts.run_safety_eval_center --only rag
    python -m scripts.run_safety_eval_center --only drift
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SUITES = ("safety", "rag", "drift")
LIVE_AGENT_MODE = False


def _run_safety() -> dict:
    from backend.services.safety_red_team import (
        DEFAULT_CSV_PATH,
        DEFAULT_OUTPUT_PATH,
        run_safety_red_team_suite,
    )

    print(f"Running safety red-team suite -> {DEFAULT_OUTPUT_PATH}")
    payload = run_safety_red_team_suite(
        output_path=DEFAULT_OUTPUT_PATH,
        csv_path=DEFAULT_CSV_PATH,
        live_agent=LIVE_AGENT_MODE,
    )
    summary = payload.get("summary") or {}
    print(
        f"  pass_rate={summary.get('pass_rate')} "
        f"failed={summary.get('failed_cases')}"
    )
    return payload


def _run_rag() -> dict:
    from backend.services.rag_eval_suite import (
        DEFAULT_CSV_PATH,
        DEFAULT_OUTPUT_PATH,
        run_rag_eval_suite,
    )

    print(f"Running RAG eval suite -> {DEFAULT_OUTPUT_PATH}")
    payload = run_rag_eval_suite(
        output_path=DEFAULT_OUTPUT_PATH,
        csv_path=DEFAULT_CSV_PATH,
        live_agent=LIVE_AGENT_MODE,
    )
    summary = payload.get("summary") or {}
    print(
        f"  pass_rate={summary.get('pass_rate')} "
        f"citation_coverage={summary.get('citation_coverage_rate')} "
        f"source_hit={summary.get('expected_source_hit_rate')} "
        f"refusal_correct={summary.get('refusal_correct_rate')} "
        f"unsafe_answer_rate={summary.get('unsafe_answer_rate')}"
    )
    return payload


def _run_drift() -> dict:
    from backend.services.drift_monitoring import DEFAULT_OUTPUT_PATH, build_drift_report

    print(f"Running drift report -> {DEFAULT_OUTPUT_PATH}")
    payload = build_drift_report(output_path=DEFAULT_OUTPUT_PATH)
    print(
        f"  data_source={payload.get('data_source')} "
        f"missing_cbc_rate={payload.get('missing_cbc_rate')} "
        f"completeness={payload.get('data_completeness_score')}"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run safety/RAG/drift evaluation suites.")
    parser.add_argument(
        "--only",
        choices=SUITES,
        action="append",
        help="Restrict to a specific suite; can be repeated.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print the full JSON summary block at the end.",
    )
    parser.add_argument(
        "--live-agent",
        action="store_true",
        help="Run slower full agent/RAG pipeline instead of deterministic offline eval mode.",
    )
    args = parser.parse_args()
    global LIVE_AGENT_MODE
    LIVE_AGENT_MODE = bool(args.live_agent)

    targets = tuple(args.only) if args.only else SUITES
    results: dict[str, dict] = {}

    if "safety" in targets:
        results["safety"] = _run_safety()
    if "rag" in targets:
        results["rag"] = _run_rag()
    if "drift" in targets:
        results["drift"] = _run_drift()

    if args.print_summary:
        condensed = {
            suite: {"summary": payload.get("summary") or payload.get("status")}
            for suite, payload in results.items()
        }
        print(json.dumps(condensed, indent=2, default=str))

    failed = []
    safety_summary = (results.get("safety") or {}).get("summary") or {}
    if safety_summary.get("status") not in (None, "passed"):
        failed.append("safety")
    rag_summary = (results.get("rag") or {}).get("summary") or {}
    if rag_summary.get("status") in ("failed",):
        failed.append("rag")

    if failed:
        print(f"\nSuites with non-passing status: {', '.join(failed)}", file=sys.stderr)
        return 1

    print("\nAll requested evaluation suites completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
