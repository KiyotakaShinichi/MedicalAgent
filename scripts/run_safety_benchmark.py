import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.safety_red_team import run_safety_red_team_suite  # noqa: E402

DEFAULT_CASES_PATH = ROOT_DIR / "benchmarks" / "safety_eval_cases.jsonl"
DEFAULT_OUTPUT_PATH = "Data/evals/safety/latest_safety_benchmark.json"
DEFAULT_CSV_PATH = "Data/evals/safety/latest_safety_benchmark.csv"


def _load_jsonl_cases(path: Path) -> list[dict]:
    if not path.exists():
        return []
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cases.append(json.loads(line))
    return cases


def main():
    parser = argparse.ArgumentParser(description="Run benchmark safety cases from benchmarks/safety_eval_cases.jsonl.")
    parser.add_argument("--cases-path", default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--live-agent", action="store_true")
    args = parser.parse_args()

    cases = _load_jsonl_cases(Path(args.cases_path))
    payload = run_safety_red_team_suite(
        output_path=args.output_path,
        csv_path=args.csv_path,
        cases=cases,
        live_agent=bool(args.live_agent),
    )

    summary = payload.get("summary") or {}
    print(json.dumps({
        "output_path": args.output_path,
        "status": summary.get("status"),
        "case_count": summary.get("total_cases"),
        "pass_rate": summary.get("pass_rate"),
        "failed_cases": summary.get("failed_cases"),
    }, indent=2))

    if summary.get("status") not in (None, "passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
