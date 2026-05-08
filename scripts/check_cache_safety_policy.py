import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["LLM_ADJUDICATION_ENABLED"] = "false"

from backend.services.agent_rag import is_cacheable, route_intent, safety_scope_check  # noqa: E402

DEFAULT_OUTPUT_PATH = ROOT_DIR / "Data" / "agent_eval" / "cache_policy_check.json"

CASES = [
    {
        "id": "education-cacheable",
        "query": "What is pCR in this project?",
        "expected_cacheable": True,
    },
    {
        "id": "portal-help-not-cacheable",
        "query": "Where can I put my CBC, medication, symptoms, and MRI uploads?",
        "expected_cacheable": False,
    },
    {
        "id": "treatment-decision-blocked",
        "query": "Based on my labs, should I delay my next chemo cycle?",
        "expected_cacheable": False,
    },
    {
        "id": "urgent-symptom-blocked",
        "query": "I have fever during chemo. What should I do?",
        "expected_cacheable": False,
    },
    {
        "id": "patient-specific-summary-blocked",
        "query": "Summarize my timeline.",
        "expected_cacheable": False,
    },
    {
        "id": "cbc-education-cacheable",
        "query": "What do CBC trends generally help monitor?",
        "expected_cacheable": True,
    },
]


def evaluate_case(case):
    query = case["query"]
    actions = case.get("actions") or []
    urgent_flags = case.get("urgent_flags") or []
    safety = safety_scope_check(query, urgent_flags)
    intent = route_intent(query, actions=actions, safety=safety)
    cacheable = is_cacheable(
        query=query,
        intent=intent,
        safety=safety,
        actions=actions,
        urgent_flags=urgent_flags,
    )
    expected = bool(case["expected_cacheable"])
    return {
        "id": case["id"],
        "query": query,
        "expected_cacheable": expected,
        "observed_cacheable": cacheable,
        "intent": intent,
        "safety_level": safety.get("level"),
        "safety_scope": safety.get("scope"),
        "passed": cacheable == expected,
    }


def main():
    results = [evaluate_case(case) for case in CASES]
    status = "passed" if all(item["passed"] for item in results) else "failed"
    output = {
        "schema_version": "cache_policy_check_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "case_count": len(results),
        "cases": results,
        "claim_boundary": "Cache safety checks are engineering policy tests, not clinical validation.",
    }
    DEFAULT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(json.dumps({
        "output_path": str(DEFAULT_OUTPUT_PATH.relative_to(ROOT_DIR)),
        "status": status,
        "failed_cases": [item["id"] for item in results if not item["passed"]],
    }, indent=2))

    if status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
