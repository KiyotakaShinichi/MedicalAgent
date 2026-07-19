import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.dependency_lock_audit import (
    write_dependency_lock_audit,
    write_environment_transitive_lock,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-transitive-lock", action="store_true")
    args = parser.parse_args()
    if args.refresh_transitive_lock:
        lock_path = write_environment_transitive_lock()
        print(f"wrote environment transitive lock: {lock_path}")
    artifact = write_dependency_lock_audit()
    print(
        "dependency lock audit:", artifact["status"],
        "complete=", artifact["lock_complete"],
        "environment match=", artifact["environment_matches_lock"],
        "transitive match=", artifact["environment_matches_transitive_lock"],
    )
