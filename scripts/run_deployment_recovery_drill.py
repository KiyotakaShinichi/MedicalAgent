from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.deployment_recovery_drill import run_local_recovery_drill


if __name__ == "__main__":
    print(json.dumps(run_local_recovery_drill(), indent=2))
