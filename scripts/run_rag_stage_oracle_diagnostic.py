"""Run the stage-wise RAG retrieval oracle diagnostic.

Writes ``Data/evals/rag/latest_rag_stage_oracle_diagnostic.json``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.rag_stage_oracle_diagnostic import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    build_report,
    write_report,
    _baseline_full_stack_recall_at_10,
)


def main() -> int:
    out = write_report(DEFAULT_OUTPUT_PATH)
    report = build_report(
        actual_full_stack_recall_at_10=_baseline_full_stack_recall_at_10(),
    )
    s = report["summary"]
    print(f"wrote: {out}")
    print(f"  total_n={s['total_n']}  corpus_coverage={s['corpus_coverage_rate']}")
    print(
        f"  candidate_recall@50  BM25={s['bm25_candidate_recall_at_50']}  "
        f"dense={s['dense_candidate_recall_at_50']}  hybrid={s['hybrid_candidate_recall_at_50']}"
    )
    print(
        f"  source_filter_retention={s['source_filter_retention_rate']}  "
        f"citation_window_retention={s['citation_window_retention_rate']}"
    )
    print(
        f"  oracle_R@10_upper_bound={s['oracle_recall_at_10_upper_bound']}  "
        f"actual_full_stack_R@10={s['actual_full_stack_recall_at_10']}  "
        f"oracle_gap={s['oracle_gap']}"
    )
    print("  failure_stage_counts:")
    for stage, n in sorted(s["failure_stage_counts"].items(), key=lambda x: -x[1]):
        print(f"    {stage:42s} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
