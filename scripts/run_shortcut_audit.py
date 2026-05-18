import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.shortcut_audit import run_shortcut_audit


if __name__ == "__main__":
    payload = run_shortcut_audit()
    print(json.dumps({
        "status": payload["status"],
        "toxicity_auc": payload["toxicity_audit"]["full_auc"],
        "dominant_shortcut_features": payload["dominant_shortcut_features"],
    }, indent=2))
    sys.exit(0 if payload["status"] in {"strong", "needs_attention"} else 1)
