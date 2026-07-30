from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.disposable_synthetic_staging_readiness import (
    collect_disposable_synthetic_runtime_observations,
    write_disposable_synthetic_staging_readiness,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="Run loopback health, restore, n8n import, and MailHog drills.",
    )
    args = parser.parse_args()
    observations = (
        collect_disposable_synthetic_runtime_observations()
        if args.runtime
        else None
    )
    result = write_disposable_synthetic_staging_readiness(
        runtime_observations=observations
    )
    validation = result["compose_validation"]
    print(
        f"status={result['status']} checks={result['passed_count']}/"
        f"{result['check_count']} docker_available={validation['available']} "
        f"runtime_started={result['runtime_started']}"
    )
