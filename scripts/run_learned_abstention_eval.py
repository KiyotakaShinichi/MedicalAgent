import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.learned_abstention import evaluate_learned_abstention


if __name__ == "__main__":
    payload = evaluate_learned_abstention()
    print(json.dumps({"status": payload["status"], "comparison": payload.get("comparison")}, indent=2))
    sys.exit(0 if payload["status"] in {"strong", "needs_attention"} else 1)
