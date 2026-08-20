import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.dep001b_candidate_freeze import freeze_dep001b_candidate


if __name__ == "__main__":
    artifact = freeze_dep001b_candidate()
    print(artifact["status"])
    print(f"frozen_artifact_count={artifact['frozen_artifact_count']}")
