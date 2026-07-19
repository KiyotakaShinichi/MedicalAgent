import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.dependency_lock_audit import write_dependency_lock_audit


if __name__ == "__main__":
    artifact = write_dependency_lock_audit()
    print(
        "dependency lock audit:", artifact["status"],
        "complete=", artifact["lock_complete"],
        "environment match=", artifact["environment_matches_lock"],
    )
