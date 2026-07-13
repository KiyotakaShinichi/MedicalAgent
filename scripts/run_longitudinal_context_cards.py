"""Emit the longitudinal context card eval artifact (eval-only)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.longitudinal_context_cards import (  # noqa: E402
    OUTPUT_PATH, build_report, write_report,
)


def main() -> int:
    write_report(OUTPUT_PATH)
    report = build_report()
    m = report.get("metrics") or {}
    print(f"wrote: {OUTPUT_PATH}")
    print(f"  n_patients={report.get('n_patients_sampled')}  n_cards={report.get('n_cards')}")
    print(f"  provenance_coverage={m.get('provenance_coverage')}  "
          f"timestamp_coverage={m.get('timestamp_coverage')}  "
          f"missing_evidence_disclosure_rate={m.get('missing_evidence_disclosure_rate')}  "
          f"unsafe_inference_rate={m.get('unsafe_inference_rate')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
