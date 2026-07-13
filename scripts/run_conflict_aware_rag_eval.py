"""Run the conflict-aware RAG adjudication eval (eval-only)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.conflict_aware_rag_adjudicator import (  # noqa: E402
    OUTPUT_PATH, build_report, write_report,
)


def main() -> int:
    write_report(OUTPUT_PATH)
    report = build_report()
    m = report.get("metrics") or {}
    print(f"wrote: {OUTPUT_PATH}")
    print(f"  n={report.get('total_n')}  status={report.get('status')}")
    print(f"  conflict_detection_rate={m.get('conflict_detection_rate')}  "
          f"conflict_resolution_rate={m.get('conflict_resolution_rate')}")
    print(f"  unsafe_consensus_rate={m.get('unsafe_consensus_rate')}  "
          f"escalation_correctness={m.get('escalation_correctness')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
