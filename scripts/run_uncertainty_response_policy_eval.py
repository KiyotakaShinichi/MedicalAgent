"""Emit the uncertainty-to-response policy eval (eval-only)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.uncertainty_response_policy_eval import (  # noqa: E402
    OUTPUT_PATH, build_report, write_report,
)


def main() -> int:
    write_report(OUTPUT_PATH)
    r = build_report()
    print(f"wrote: {OUTPUT_PATH}")
    print(f"  pass_rate={r['metrics']['pass_rate']}  "
          f"unsafe_route_rate={r['metrics']['unsafe_route_rate']}  "
          f"policy_coverage={r['metrics']['policy_coverage']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
