"""Run the citation-precision failure analysis.

Writes ``Data/evals/rag/latest_citation_precision_failure_analysis.json``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.citation_precision_failure_analysis import (  # noqa: E402
    OUTPUT_PATH,
    build_report,
    write_report,
)


def main() -> int:
    out = write_report(OUTPUT_PATH)
    report = build_report()
    print(f"wrote: {out}")
    print(f"  total full-stack low_citation_precision failures: {report['total_n']}")
    for cat, n in sorted(report["category_counts"].items(), key=lambda x: -x[1]):
        if n > 0:
            print(f"  {cat:50s} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
