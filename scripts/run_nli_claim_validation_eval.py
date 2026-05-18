from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.rag_claim_validator import validate_claims


CASES_PATH = ROOT / "evals" / "rag_claim_validation_cases.json"
OUTPUT_PATH = ROOT / "Data" / "evals" / "rag" / "latest_nli_claim_validation_eval.json"


def main() -> int:
    doc = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    hard_failures: list[dict[str, object]] = []

    for case in doc.get("cases", []):
        result = validate_claims(case["reply"], case.get("chunks") or [], method="nli")
        verdicts = [v.to_dict() for v in result.verdicts if v.is_claim]
        observed = verdicts[0] if verdicts else {"status": "non_claim"}
        nli_available = bool(result.nli_available)
        expected = case.get("expected_status")
        passed = observed.get("status") == expected
        row = {
            "id": case["id"],
            "expected_status": expected,
            "observed_status": observed.get("status"),
            "observed_reason": observed.get("reason"),
            "validation_method": result.validation_method,
            "nli_required": bool(case.get("nli_required")),
            "nli_available": nli_available,
            "entailment_score": observed.get("entailment_score"),
            "contradiction_score": observed.get("contradiction_score"),
            "passed": passed,
        }
        rows.append(row)
        if nli_available and not passed:
            hard_failures.append(row)

    nli_available_cases = sum(1 for row in rows if row["nli_available"])
    if nli_available_cases == 0:
        status = "optional_unavailable"
    elif hard_failures:
        status = "needs_attention"
    else:
        status = "strong"

    payload = {
        "schema_version": "nli_claim_validation_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "method": "nli_entailment_optional",
        "summary": {
            "case_count": len(rows),
            "passed_when_nli_available": sum(1 for row in rows if row["nli_available"] and row["passed"]),
            "hard_failures": len(hard_failures),
            "nli_required_cases": sum(1 for row in rows if row["nli_required"]),
            "nli_available_cases": nli_available_cases,
        },
        "cases": rows,
        "setup": {
            "install": "pip install transformers torch",
            "run": "python scripts/run_nli_claim_validation_eval.py",
            "env": {
                "ONCOTRACK_RAG_CLAIM_VALIDATOR": "nli",
                "ONCOTRACK_RAG_NLI_MODEL": "typeform/distilbert-base-uncased-mnli",
                "ONCOTRACK_NLI_ALLOW_DOWNLOAD": "1",
            },
            "default_behavior": (
                "Uses only a locally cached NLI model by default. Set "
                "ONCOTRACK_NLI_ALLOW_DOWNLOAD=1 when you deliberately want "
                "the script to download the model."
            ),
        },
        "claim_boundary": (
            "Optional local NLI/entailment validation is an engineering guardrail. "
            "It is not a clinical fact checker and remains outside the hard CI path "
            "unless the dependency and model are explicitly installed."
        ),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "summary": payload["summary"]}, indent=2))
    # Optional-unavailable is an honest local state, not a CI failure.
    return 0 if status in {"strong", "optional_unavailable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
