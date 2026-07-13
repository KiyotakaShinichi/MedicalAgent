"""Emit the external-red-team readiness artifact (no fabrication)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.external_red_team_readiness import (  # noqa: E402
    OUTPUT_PATH, build_readiness, write_readiness,
)


def main() -> int:
    write_readiness(OUTPUT_PATH)
    r = build_readiness()
    print(f"wrote: {OUTPUT_PATH}")
    print(f"  status={r['status']}  completed_external_cases={r['completed_external_cases']}  "
          f"template={r['template_present']}  quickstart={r['quickstart_present']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
