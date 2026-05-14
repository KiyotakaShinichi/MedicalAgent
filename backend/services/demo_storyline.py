from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from backend.database import SessionLocal
from backend.models import ImagingReport, LabResult, Patient, SymptomReport, Treatment


DEFAULT_OUTPUT_DIR = "Data/demo_storyline"


def build_demo_storyline(patient_id: str = "P001", output_dir: str = DEFAULT_OUTPUT_DIR) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    db = SessionLocal()
    try:
        patient = db.query(Patient).filter(Patient.id == patient_id).first()
        labs = db.query(LabResult).filter(LabResult.patient_id == patient_id).order_by(LabResult.date.asc()).all()
        symptoms = (
            db.query(SymptomReport)
            .filter(SymptomReport.patient_id == patient_id)
            .order_by(SymptomReport.date.asc())
            .all()
        )
        treatments = db.query(Treatment).filter(Treatment.patient_id == patient_id).order_by(Treatment.date.asc()).all()
        imaging = (
            db.query(ImagingReport)
            .filter(ImagingReport.patient_id == patient_id)
            .order_by(ImagingReport.date.asc())
            .all()
        )
    finally:
        db.close()

    scenes = [
        {
            "step": 1,
            "title": "Login and patient-scoped record",
            "what_to_show": "Login as P001 and confirm the dashboard only exposes that patient journey.",
            "evidence": {
                "patient_id": patient_id,
                "patient_name": getattr(patient, "name", None) if patient else None,
                "diagnosis": getattr(patient, "diagnosis", None) if patient else None,
            },
            "talk_track": "Role isolation and patient scoping are part of the safety story.",
        },
        {
            "step": 2,
            "title": "Longitudinal treatment timeline",
            "what_to_show": "Scroll the timeline from treatment cycles to labs, symptoms, and imaging.",
            "evidence": {
                "treatment_count": len(treatments),
                "lab_count": len(labs),
                "symptom_count": len(symptoms),
                "imaging_count": len(imaging),
            },
            "talk_track": "The central object is a patient timeline, not a generic chatbot transcript.",
        },
        {
            "step": 3,
            "title": "Deterministic safety before LLM",
            "what_to_show": "Point at CBC and symptom risk flags before opening the support agent.",
            "evidence": _latest_labs(labs),
            "talk_track": "CBC and symptom thresholds generate review flags before any language model response.",
        },
        {
            "step": 4,
            "title": "Support agent routing",
            "what_to_show": "Ask a casual question, a RAG education question, then provide a clear symptom or lab record.",
            "evidence": {
                "casual_prompt": "hi, who are you?",
                "rag_prompt": "what does pCR mean in breast cancer monitoring?",
                "tool_prompt": "nausea severity 6/10 today",
            },
            "talk_track": "The agent routes between conversation, grounded education, and explicit data-entry tools.",
        },
        {
            "step": 5,
            "title": "Clinician review and audit trail",
            "what_to_show": "Open clinician queue, select the patient, approve/edit/reject the AI summary.",
            "evidence": {
                "queue_reason": "risk flags and monitoring score drive the review queue",
            },
            "talk_track": "The system surfaces signals; the clinician remains the decision-maker.",
        },
        {
            "step": 6,
            "title": "Admin/MLE governance",
            "what_to_show": "Open eval panels: RAG ablation, agent traces, MLE readiness, calibration, errors, latency.",
            "evidence": {
                "artifact_paths": [
                    "Data/mle_monitoring/latest_mle_readiness.json",
                    "Data/evals/agent_regression/latest_agent_regression.json",
                    "Data/evals/rag_gold/latest_rag_gold_report.json",
                    "Data/evals/narrative/latest_ai_ml_eval_narrative.md",
                ],
            },
            "talk_track": "The portfolio value is the observable AI lifecycle, not just the final answer.",
        },
    ]

    report = {
        "schema_version": "demo_storyline_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "patient_id": patient_id,
        "scenes": scenes,
        "demo_prompts": [scene["evidence"] for scene in scenes if scene["step"] == 4],
        "claim_boundary": "Synthetic demo only. The platform is non-diagnostic and requires clinician review.",
    }
    json_path = output_path / f"{patient_id}_storyline.json"
    md_path = output_path / f"{patient_id}_storyline.md"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    report["files"] = {"json": str(json_path), "markdown": str(md_path)}
    return report


def _latest_labs(labs) -> dict:
    if not labs:
        return {"latest": None}
    row = labs[-1]
    return {
        "latest_date": str(getattr(row, "date", None)),
        "wbc": getattr(row, "wbc", None),
        "hemoglobin": getattr(row, "hemoglobin", None),
        "platelets": getattr(row, "platelets", None),
    }


def _markdown(report: dict) -> str:
    lines = [
        f"# Demo Storyline: {report['patient_id']}",
        "",
        "Use this as a repeatable walkthrough for the patient, clinician, and admin surfaces.",
        "",
    ]
    for scene in report["scenes"]:
        lines.extend([
            f"## {scene['step']}. {scene['title']}",
            f"- Show: {scene['what_to_show']}",
            f"- Say: {scene['talk_track']}",
            f"- Evidence: `{json.dumps(scene['evidence'], default=str)}`",
            "",
        ])
    lines.extend(["## Claim Boundary", "", report["claim_boundary"], ""])
    return "\n".join(lines)
