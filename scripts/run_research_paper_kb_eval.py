from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.research_paper_kb_eval import (  # noqa: E402
    run_research_paper_kb_eval,
)


if __name__ == "__main__":
    reports = run_research_paper_kb_eval()
    audit = reports["audit"]
    evaluation = reports["evaluation"]

    if not evaluation.get("evaluated", True):
        # The optional corpus is not here. Say so loudly and exit clean: the
        # five research-paper artifacts are `required: false`, so the release
        # gate treats a non-result as a warning rather than a blocker.
        print(
            "Research-paper KB evaluation: "
            f"{evaluation['status'].upper()} - {evaluation['reason']}"
        )
        print(
            "  no metrics were produced, and the previous run's numbers were "
            "overwritten so they cannot be read as current evidence."
        )
        raise SystemExit(0)

    summary = evaluation["summary"]
    print(
        f"status={evaluation['status']} papers={evaluation['paper_count']} "
        f"cases={evaluation['case_count']} "
        f"full_stack_recall_at_10={summary['full_stack_recall_at_10']} "
        f"top1={summary['full_stack_top1_paper_accuracy']} "
        f"false_pmcid_identity={audit['summary']['false_pmcid_identity_chunk_count']} "
        f"improvement_proven_vs_bm25="
        f"{summary['paper_retrieval_improvement_proven_vs_bm25']}"
    )
