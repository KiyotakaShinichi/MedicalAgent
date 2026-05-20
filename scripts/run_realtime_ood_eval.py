from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.realtime_ood_gate import run_realtime_ood_eval


if __name__ == "__main__":
    payload = run_realtime_ood_eval()
    print(payload["summary"])
