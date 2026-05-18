import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.synthetic_realism_hardening import build_synthetic_realism_hardening_report


if __name__ == "__main__":
    payload = build_synthetic_realism_hardening_report()
    print(json.dumps({"status": payload["status"], **payload["summary"]}, indent=2))
    sys.exit(0 if payload["status"] in {"strong", "needs_attention"} else 1)
