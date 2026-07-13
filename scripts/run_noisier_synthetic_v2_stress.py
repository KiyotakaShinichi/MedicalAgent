"""Run the noisier synthetic v2 stress benchmark.

Scaffold-only.  Writes
``Data/evals/models/latest_noisier_synthetic_v2_stress.json``.
Does NOT retrain any model and does NOT change live inference
defaults.  Promotion decision is always ``reject_or_hold``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.noisier_synthetic_v2_stress import (  # noqa: E402
    OUTPUT_PATH,
    build_report,
    write_report,
)


def main() -> int:
    out = write_report(OUTPUT_PATH)
    report = build_report()
    print(f"wrote: {out}")
    print(f"  status={report['status']}  global_promotion_decision={report['global_promotion_decision']}")
    if report.get("clean_metrics") is None:
        return 0
    print(f"  clean_metrics: {report['clean_metrics']}")
    for noise in report.get("per_noise_type", []):
        print(
            f"  {noise['noise_type']:32s} leakage_status={noise['leakage_status']}  "
            f"deltas={noise['deltas']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
