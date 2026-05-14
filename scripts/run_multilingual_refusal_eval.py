import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.agent_rag import route_intent, safety_scope_check  # noqa: E402
from backend.services.artifact_manifest import build_artifact_manifest  # noqa: E402


DEFAULT_CASES_PATH = "evals/multilingual_refusal_cases.json"
DEFAULT_OUTPUT_PATH = "Data/evals/safety/latest_multilingual_refusal_eval.json"


def main():
    parser = argparse.ArgumentParser(description="Run multilingual refusal routing checks.")
    parser.add_argument("--cases-path", default=DEFAULT_CASES_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    report = run_eval(args.cases_path, args.output_path)
    print(json.dumps({
        "output_path": args.output_path,
        "status": report["summary"]["status"],
        "pass_rate": report["summary"]["pass_rate"],
        "case_count": report["summary"]["case_count"],
    }, indent=2))


def run_eval(cases_path=DEFAULT_CASES_PATH, output_path=DEFAULT_OUTPUT_PATH):
    catalog = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    rows = []
    for case in catalog.get("cases") or []:
        query = case.get("query") or ""
        safety = safety_scope_check(query)
        intent = route_intent(query, safety=safety)
        passed = (
            safety.get("scope") == case.get("expected_scope")
            and intent == case.get("expected_intent")
        )
        rows.append({
            "case_id": case.get("id"),
            "language": case.get("language"),
            "query": query,
            "expected_scope": case.get("expected_scope"),
            "observed_scope": safety.get("scope"),
            "expected_intent": case.get("expected_intent"),
            "observed_intent": intent,
            "pass": passed,
        })

    passed_count = sum(1 for row in rows if row["pass"])
    report = {
        **build_artifact_manifest(dataset_paths={"multilingual_refusal_cases": cases_path}),
        "schema_version": "multilingual_refusal_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Tagalog/code-switched diagnosis, treatment-decision, and urgent-symptom routing regression.",
        "summary": {
            "status": "strong" if passed_count == len(rows) else "needs_attention",
            "case_count": len(rows),
            "passed": passed_count,
            "pass_rate": round(passed_count / len(rows), 4) if rows else None,
            "failed_cases": [row["case_id"] for row in rows if not row["pass"]],
        },
        "cases": rows,
        "limitations": [
            "Small deterministic set focused on Tagalog/Taglish probes.",
            "Does not prove open-ended multilingual safety; use live LLM/adversarial evals for deeper review.",
        ],
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
