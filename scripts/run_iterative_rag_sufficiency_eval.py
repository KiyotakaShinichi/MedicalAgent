"""Run the iterative RAG sufficiency eval scaffold.

Writes ``Data/evals/rag/latest_iterative_rag_sufficiency_eval.json``.
Eval-only; live agent unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.iterative_rag_sufficiency import (  # noqa: E402
    OUTPUT_PATH, build_report, write_report,
)


def main() -> int:
    out = write_report(OUTPUT_PATH)
    report = build_report()
    print(f"wrote: {out}")
    m = report.get("metrics") or {}
    print(f"  n={report.get('total_n')}  status={report.get('status')}")
    print(f"  initial_answerability={m.get('initial_answerability_rate')}  "
          f"second_pass={m.get('second_pass_answerability_rate')}  "
          f"insufficiency_reduction={m.get('insufficiency_reduction_rate')}")
    print(f"  unsafe_answer_rate={m.get('unsafe_answer_rate')}  "
          f"latency_delta_ms={m.get('latency_delta_ms')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
