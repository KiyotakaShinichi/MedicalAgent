import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.durable_automation_worker_eval import build_durable_automation_worker_eval


if __name__ == "__main__":
    result = build_durable_automation_worker_eval()
    print(f"status={result['status']} control_pass_rate={result['control_pass_rate']}")
