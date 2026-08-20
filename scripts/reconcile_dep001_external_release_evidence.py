from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.dep001_release_evidence import build_release_compatible_evidence


SOURCE = ROOT / "Data/evals/safety/dep001a/latest_external_holdout_evaluation.json"
TARGET = ROOT / "Data/evals/safety/latest_dep001_safety_assurance.json"


def main() -> int:
    source_bytes = SOURCE.read_bytes()
    canonical = json.loads(source_bytes.decode("utf-8"))
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    derived = build_release_compatible_evidence(
        canonical,
        source_path=str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        source_sha256=source_sha256,
    )
    TARGET.write_text(json.dumps(derived, indent=2) + "\n", encoding="utf-8")
    print(f"source_sha256={source_sha256}")
    print(f"status={derived['status']}")
    print(f"dep001_complete={str(derived['dep001_complete']).lower()}")
    print("compatibility_aliases_only=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
