from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.toxicity_shortcut_audit import run_toxicity_shortcut_audit


if __name__ == "__main__":
    report = run_toxicity_shortcut_audit()
    print(json.dumps({
        "status": report["status"],
        "positive_label_rate": report["positive_label_rate"],
        "rule_accuracy": report["rule_reconstruction"]["accuracy"],
        "rule_auroc": report["rule_reconstruction"]["auroc"],
        "recommended_use": report["recommendation"]["use"],
    }, indent=2))
