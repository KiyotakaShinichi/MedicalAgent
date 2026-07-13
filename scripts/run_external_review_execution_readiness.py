"""Emit the external-review execution readiness artifact.

Counts attestation files actually committed in
``Data/evals/external_review/`` and reports whether the outreach +
intake infrastructure is in place.  Does NOT fabricate any review.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.external_review_execution_readiness import (  # noqa: E402
    OUTPUT_PATH,
    build_readiness,
    write_readiness,
)


def main() -> int:
    out = write_readiness(OUTPUT_PATH)
    report = build_readiness()
    print(f"wrote: {out}")
    print(f"  status:             {report['status']}")
    print(f"  completed_reviews:  {report['completed_reviews']}")
    print(f"  prepared_templates: {len(report['prepared_templates'])}")
    print(f"  prepared_packets:   {len(report['prepared_packets'])}")
    print(f"  next_best_reviewer: {report['next_best_reviewer']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
