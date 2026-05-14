from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from backend.services.agent_rag import route_intent, safety_scope_check
from backend.services.artifact_manifest import build_artifact_manifest


DEFAULT_CASES_PATH = "evals/multilingual_refusal_cases.json"
DEFAULT_OUTPUT_PATH = "Data/evals/safety/latest_multilingual_refusal_eval.json"


def run_multilingual_refusal_eval(
    cases_path: str = DEFAULT_CASES_PATH,
    output_path: str | None = DEFAULT_OUTPUT_PATH,
) -> dict:
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
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def load_multilingual_refusal_eval(path: str = DEFAULT_OUTPUT_PATH) -> dict:
    artifact = Path(path)
    if not artifact.exists():
        return {
            "schema_version": "multilingual_refusal_eval_v1",
            "status": "not_generated",
            "message": "No multilingual refusal report has been generated yet.",
        }
    try:
        return json.loads(artifact.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "schema_version": "multilingual_refusal_eval_v1",
            "status": "error",
            "message": str(exc),
        }
