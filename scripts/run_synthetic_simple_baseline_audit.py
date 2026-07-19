import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.synthetic_simple_baseline_audit import write_simple_baseline_audit


if __name__ == "__main__":
    artifact = write_simple_baseline_audit()
    comparison = artifact["paired_champion_vs_logistic"]
    print(
        "synthetic simple-baseline audit:",
        artifact["status"],
        "n=", artifact["total_n"],
        "champion superiority over logistic=", comparison["superiority_proven"],
    )
