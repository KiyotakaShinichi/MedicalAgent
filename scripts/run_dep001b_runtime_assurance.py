import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.dep001b_runtime_assurance import build_dep001b_runtime_assurance


if __name__ == "__main__":
    artifact = build_dep001b_runtime_assurance()
    print(artifact["status"])
    print(artifact["metrics"])
    print(artifact["fault_injection"])
