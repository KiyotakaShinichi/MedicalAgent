from __future__ import annotations

import argparse
import json

from .runner import run_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reproducible NLCare engineering evaluations.")
    for flag in ("quick", "full", "retrieval", "safety", "security", "ml", "rag", "automation", "xai", "load", "saas"):
        parser.add_argument(f"--{flag}", action="store_true")
    args = parser.parse_args()
    suites = {name for name, enabled in vars(args).items() if enabled}
    result = run_evaluation(suites or {"quick"})
    print(json.dumps({"status": result["status"], "summary": result["summary"]}, indent=2))
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
