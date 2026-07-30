from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.disposable_synthetic_staging_readiness import (
    write_disposable_synthetic_staging_readiness,
)


if __name__ == "__main__":
    result = write_disposable_synthetic_staging_readiness()
    validation = result["compose_validation"]
    print(
        f"status={result['status']} checks={result['passed_count']}/"
        f"{result['check_count']} docker_available={validation['available']} "
        f"runtime_started={result['runtime_started']}"
    )
