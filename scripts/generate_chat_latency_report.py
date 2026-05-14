import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.chat_latency_report import build_chat_latency_report  # noqa: E402


def main():
    report = build_chat_latency_report()
    print(json.dumps({
        "status": report.get("status"),
        "sample_size": report.get("sample_size"),
        "p50_latency_ms": report.get("p50_latency_ms"),
        "p95_latency_ms": report.get("p95_latency_ms"),
        "output_path": "Data/evals/latency/latest_chat_latency_report.json",
    }, indent=2))


if __name__ == "__main__":
    main()
