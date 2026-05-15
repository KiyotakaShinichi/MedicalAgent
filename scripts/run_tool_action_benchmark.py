from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.database import SessionLocal  # noqa: E402
from backend.models import ImagingReport, LabResult, SymptomReport  # noqa: E402
from backend.services.support_chat_agent import handle_patient_chat  # noqa: E402


DEFAULT_OUTPUT_PATH = ROOT_DIR / "Data/evals/tool_actions/latest_tool_action_benchmark.json"
PATIENT_ID = "P001"
MARKER = "tool-action-benchmark"


CASES = [
    {
        "id": "save_symptom_bloody_discharge",
        "message": "I have bloody discharge severity 9/10 on May 7th tool-action-benchmark",
        "expected_action": "saved_symptom",
        "expected_text": ["bloody discharge", "9"],
    },
    {
        "id": "save_complete_cbc",
        "message": "CBC today: WBC 3.1, hemoglobin 10.5, platelets 120 tool-action-benchmark",
        "expected_action": "saved_labs",
        "expected_text": ["CBC"],
    },
    {
        "id": "save_mri_report",
        "message": "MRI 2026-05-07 findings tool-action-benchmark right breast mass decreased impression partial response",
        "expected_action": "saved_imaging_report",
        "expected_text": ["MRI", "2026-05-07"],
    },
    {
        "id": "save_ct_report_metastatic_indicator",
        "message": "CT abdomen 2026-05-07 findings tool-action-benchmark ascites and liver lesion impression possible metastatic indicator",
        "expected_action": "saved_imaging_report",
        "expected_secondary_action": "possible_metastatic_indicator",
        "expected_text": ["CT", "2026-05-07"],
    },
    {
        "id": "save_ultrasound_report",
        "message": "Ultrasound May 7 findings tool-action-benchmark breast lesion stable impression BI-RADS 3",
        "expected_action": "saved_imaging_report",
        "expected_text": ["ultrasound", "2026-05-07"],
    },
    {
        "id": "partial_symptom_requires_severity",
        "message": "I have nausea tool-action-benchmark",
        "expected_action": "partial_symptom_detected",
        "expected_text": ["severity", "0-10"],
    },
]


def main() -> int:
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT_PATH
    db = SessionLocal()
    try:
        _cleanup(db)
        rows = []
        for case in CASES:
            started = time.perf_counter()
            result = handle_patient_chat(db, PATIENT_ID, case["message"])
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            actions = result.get("saved_actions") or []
            action_types = [action.get("type") for action in actions]
            reply = result.get("reply") or ""
            passed = case["expected_action"] in action_types and all(
                text.lower() in reply.lower() for text in case.get("expected_text", [])
            )
            if case.get("expected_secondary_action"):
                passed = passed and case["expected_secondary_action"] in action_types
            rows.append({
                "id": case["id"],
                "status": "passed" if passed else "failed",
                "latency_ms": latency_ms,
                "action_types": action_types,
                "reply_preview": reply[:240],
                "expected_action": case["expected_action"],
            })

        passed_count = sum(1 for row in rows if row["status"] == "passed")
        latencies = [row["latency_ms"] for row in rows]
        payload = {
            "schema_version": "tool_action_benchmark_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if passed_count == len(rows) else "failed",
            "summary": {
                "case_count": len(rows),
                "passed": passed_count,
                "pass_rate": round(passed_count / len(rows), 3),
                "max_latency_ms": max(latencies) if latencies else None,
                "average_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
            },
            "cases": rows,
            "claim_boundary": (
                "This benchmark verifies deterministic patient-support tool saves and latency. "
                "It does not validate clinical correctness."
            ),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"status": payload["status"], "summary": payload["summary"], "output_path": str(output_path)}, indent=2))
        return 0 if payload["status"] == "passed" else 1
    finally:
        _cleanup(db)
        db.close()


def _cleanup(db) -> None:
    db.query(SymptomReport).filter(
        SymptomReport.patient_id == PATIENT_ID,
        SymptomReport.notes.like(f"%{MARKER}%"),
    ).delete(synchronize_session=False)
    db.query(LabResult).filter(
        LabResult.patient_id == PATIENT_ID,
        LabResult.source_note.like(f"%{MARKER}%"),
    ).delete(synchronize_session=False)
    db.query(ImagingReport).filter(
        ImagingReport.patient_id == PATIENT_ID,
        ImagingReport.findings.like(f"%{MARKER}%"),
    ).delete(synchronize_session=False)
    db.commit()


if __name__ == "__main__":
    raise SystemExit(main())
