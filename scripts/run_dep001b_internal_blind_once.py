import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.dep001b_internal_blind_evaluation import run_dep001b_internal_blind_once


if __name__ == "__main__":
    artifact = run_dep001b_internal_blind_once()
    print(artifact["status"])
    print(artifact["metrics"])
