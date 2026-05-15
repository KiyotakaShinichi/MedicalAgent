import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.safety_red_team import run_safety_red_team_suite  # noqa: E402

DEFAULT_CASES_PATH = ROOT_DIR / "benchmarks" / "adversarial_eval_cases.jsonl"
DEFAULT_OUTPUT_PATH = "Data/evals/safety/latest_adversarial_eval.json"
DEFAULT_CSV_PATH = "Data/evals/safety/latest_adversarial_eval.csv"


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
    parser = argparse.ArgumentParser(description="Run adversarial safety benchmark cases.")
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

    # Augment the suite report with per-category attack-block recall so a
    # reviewer can read "did we block X% of prompt-injection attacks" rather
    # than just a global pass_rate.  Recall is computed per category as
    # (passed_cases_in_category / total_cases_in_category).
    per_category = _per_category_recall(payload.get("cases") or [])
    payload.setdefault("by_category", per_category)
    payload.setdefault("category_attack_block_recall",
                       {cat: cat_stats["recall"] for cat, cat_stats in per_category.items()})
    Path(args.output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary = payload.get("summary") or {}
    print(json.dumps({
        "output_path": args.output_path,
        "status": summary.get("status"),
        "case_count": summary.get("total_cases"),
        "pass_rate": summary.get("pass_rate"),
        "failed_cases": summary.get("failed_cases"),
        "category_attack_block_recall": payload.get("category_attack_block_recall"),
    }, indent=2))

    if summary.get("status") not in (None, "passed"):
        raise SystemExit(1)


def _per_category_recall(cases: list[dict]) -> dict:
    """Aggregate per-category counts + recall from a flat list of case rows.

    The suite uses ``case["pass"]`` (boolean) as the per-case verdict.
    Recall = passed / total within each adversarial category.
    """
    buckets: dict[str, dict[str, int]] = {}
    for case in cases:
        category = str(case.get("category") or "uncategorized")
        # The suite emits ``pass`` (true/false). Some legacy callers used
        # ``passed`` — accept both so a schema change doesn't silently break.
        passed = bool(case.get("pass") if "pass" in case else case.get("passed"))
        bucket = buckets.setdefault(category, {"passed": 0, "total": 0})
        bucket["total"] += 1
        if passed:
            bucket["passed"] += 1
    return {
        category: {
            **bucket,
            "recall": round(bucket["passed"] / bucket["total"], 3) if bucket["total"] else None,
        }
        for category, bucket in buckets.items()
    }


if __name__ == "__main__":
    main()
