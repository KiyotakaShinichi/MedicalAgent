"""Run the synthetic-only timeline drift stress scaffold (review-only)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.synthetic_timeline_drift_stress import (  # noqa: E402
    OUTPUT_PATH, build_report, write_report,
)


def main() -> int:
    write_report(OUTPUT_PATH)
    r = build_report()
    m = r.get("metrics") or {}
    print(f"wrote: {OUTPUT_PATH}")
    print(f"  distribution_shift_detection_rate={m.get('distribution_shift_detection_rate')}  "
          f"false_shift_rate_on_baseline_synthetic={m.get('false_shift_rate_on_baseline_synthetic')}")
    print(f"  lab_trend={m.get('lab_trend_shift_detection')}  "
          f"symptom_trend={m.get('symptom_trend_shift_detection')}  "
          f"missingness={m.get('missingness_shift_detection')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
