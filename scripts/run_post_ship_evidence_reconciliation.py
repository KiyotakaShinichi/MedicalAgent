"""Reconcile reviewer-facing evidence against a completed release manifest."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECONCILIATION_SCRIPTS = (
    "scripts/run_evidence_maturity_matrix.py",
    "scripts/run_release_decision_surface.py",
    "scripts/run_focused_release_summary.py",
    "scripts/run_senior_engineering_evidence.py",
    "scripts/run_credibility_gap_registry.py",
    "scripts/run_constraint_aware_improvement_program.py",
    "scripts/run_cross_domain_assurance_eval.py",
    "scripts/run_ops_health_snapshot.py",
    "scripts/run_focused_release_summary.py",
    "scripts/run_release_gate.py",
)


def main() -> int:
    for script in RECONCILIATION_SCRIPTS:
        print(f"[post-ship] {script}", flush=True)
        subprocess.run(
            [sys.executable, script],
            cwd=ROOT,
            check=True,
            timeout=240,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
