from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.container_recovery_smoke import run_container_recovery_smoke


if __name__ == "__main__":
    result = run_container_recovery_smoke()
    print(f"status={result['status']} completed={result['completed']} docker={result['docker_available']}")
    raise SystemExit(0 if result["completed"] or result["status"] == "blocked_environment" else 1)
