"""Emit the deployment-boundary check artifact.

Writes ``Data/evals/ops/latest_deployment_boundary_check.json``.

``status_label`` is permanently ``production_shaped_not_healthcare_production_ready``
and the test suite enforces it.  No HIPAA / clinical-deployment claim
is implied by a passing run.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.deployment_boundary_check import (  # noqa: E402
    OUTPUT_PATH,
    build_report,
    write_report,
)


def main() -> int:
    out = write_report(OUTPUT_PATH)
    report = build_report()
    print(f"wrote: {out}")
    print(f"  status:        {report['status']}  ({report['status_label']})")
    print(f"  checks passed: {report['n_passed']}/{report['n_checks']}")
    for r in report["failed_checks"]:
        print(f"  FAIL  {r['name']}: {r.get('reason') or 'see evidence_path'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
