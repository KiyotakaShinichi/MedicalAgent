from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.ops_health_snapshot import build_service_health_snapshot


if __name__ == "__main__":
    report = build_service_health_snapshot()
    print(report["status"])
