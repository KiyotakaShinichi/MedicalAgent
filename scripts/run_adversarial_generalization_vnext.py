import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.adversarial_generalization_vnext import build_adversarial_generalization_vnext  # noqa: E402


if __name__ == "__main__":
    report = build_adversarial_generalization_vnext()
    print(json.dumps({
        "status": report["status"],
        "v7_source_pass_rate": report["frozen_v7_read_only_attribution"]["source_pass_rate"],
        "mutation_pass_rate": report["mutation_matrix"]["pass_rate"],
        "safe_negative_pass_rate": report["mutation_matrix"]["safe_negative_pass_rate"],
        "external_generalization_status": report["external_generalization_status"],
    }, indent=2))
