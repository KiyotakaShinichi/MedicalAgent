"""Emit the claim-source alignment hardening ledger.

Read-only enrichment of ``latest_claim_source_alignment_eval.json``.
Writes ``Data/evals/rag/latest_claim_source_alignment_hardening.json``.

Does NOT claim clinical-grade entailment.  Validator method reflects
``ONCOTRACK_RAG_CLAIM_VALIDATOR`` (default heuristic).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.claim_source_alignment_hardening import (  # noqa: E402
    OUTPUT_PATH,
    build_report,
    write_report,
)


def main() -> int:
    out = write_report(OUTPUT_PATH)
    report = build_report()
    print(f"wrote: {out}")
    print(f"  validator_method:               {report.get('validator_method')}")
    print(f"  n_rows:                         {report.get('n_rows')}")
    print(f"  patient_facing_allowed_rate:    {report.get('patient_facing_allowed_rate')}")
    print(f"  support_status_counts:          {report.get('support_status_counts')}")
    print(f"  contradiction_category_counts:  {report.get('contradiction_category_counts')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
