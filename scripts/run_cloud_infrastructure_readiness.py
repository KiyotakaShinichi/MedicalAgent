from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.cloud_infrastructure_readiness import build_cloud_infrastructure_readiness


if __name__ == "__main__":
    print(json.dumps(build_cloud_infrastructure_readiness(), indent=2))
