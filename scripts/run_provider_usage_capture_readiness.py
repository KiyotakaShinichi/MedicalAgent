from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.provider_usage_capture_readiness import (
    write_provider_usage_capture_readiness,
)


if __name__ == "__main__":
    result = write_provider_usage_capture_readiness()
    print(
        f"status={result['status']} checks={result['passed_count']}/"
        f"{result['check_count']} provider_configured="
        f"{result['provider_credentials_configured']} paired="
        f"{result['paired_request_count']}/{result['required_paired_request_count']}"
    )
