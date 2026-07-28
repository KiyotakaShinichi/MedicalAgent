from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.managed_vector_shadow_comparison import (
    build_managed_vector_shadow_comparison,
)


def main() -> int:
    report = build_managed_vector_shadow_comparison()
    print(json.dumps(report, indent=2))
    return 0 if report["status"] != "needs_attention" else 1


if __name__ == "__main__":
    raise SystemExit(main())
