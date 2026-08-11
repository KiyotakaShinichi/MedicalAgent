from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.saas_foundation_readiness import write_saas_foundation_readiness  # noqa: E402


if __name__ == "__main__":
    result = write_saas_foundation_readiness()
    print(
        f"status={result['status']} controls={result['passed_control_count']}/"
        f"{result['control_count']} clinical_validation={result['clinical_validation']}"
    )
