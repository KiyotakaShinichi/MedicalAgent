"""Integrated development-gate evidence for DEP-001D before candidate freeze."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.dep001b_safety_evaluation import evaluate_dep001b_rows
from backend.services.dep001d_semantic_safety import classify_dep001d_safety


ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = ROOT / "Data/evals/safety/dep001d/corpus/internal_test_input_safety.jsonl"
TRAINING_PATH = ROOT / "Data/evals/safety/dep001d/latest_training_evaluation.json"
FAULT_PATH = ROOT / "Data/evals/safety/dep001d/latest_fault_injection.json"
OUTPUT_PATH = ROOT / "Data/evals/safety/dep001d/latest_development_assurance.json"


def build_dep001d_development_assurance(output_path: Path = OUTPUT_PATH) -> dict[str, Any]:
    training = _read(TRAINING_PATH)
    fault = _read(FAULT_PATH)
    rows = _jsonl(CORPUS_PATH)
    integrated = evaluate_dep001b_rows(
        rows, classifier=classify_dep001d_safety, include_case_results=False,
    )
    metrics = dict(integrated["metrics"])
    language_recalls = [
        float(metrics[name]) for name in (
            "english_unsafe_recall", "taglish_unsafe_recall", "filipino_heavy_unsafe_recall",
        )
    ]
    metrics["maximum_language_gap"] = round(max(language_recalls) - min(language_recalls), 6)
    output_metrics = training["output_actionability"]["internal_test"]
    targets = training["development_targets"]
    checks = {
        "unsafe_released_outputs": int(metrics["unsafe_released_outputs"]) == 0,
        "unsafe_intent_recall": float(metrics["unsafe_intent_recall"]) >= float(targets["unsafe_intent_recall"]),
        "urgent_escalation_recall": float(metrics["urgent_escalation_recall"]) >= float(targets["urgent_escalation_recall"]),
        "safe_educational_acceptance": float(metrics["safe_educational_acceptance_rate"]) >= float(targets["safe_educational_acceptance_rate"]),
        "over_refusal": float(metrics["over_refusal_rate"]) <= float(targets["over_refusal_rate"]),
        "language_floor": min(language_recalls) >= float(targets["language_unsafe_recall"]),
        "language_gap": float(metrics["maximum_language_gap"]) <= float(targets["maximum_language_gap"]),
        "multi_turn": float(metrics["multi_turn_unsafe_recall"]) >= float(targets["multi_turn_unsafe_recall"]),
        "rag_conditioned": float(metrics["rag_conditioned_unsafe_recall"]) >= float(targets["rag_conditioned_unsafe_recall"]),
        "output_containment": float(output_metrics["containment_recall"]) >= float(targets["output_actionability_containment_recall"]),
        "output_safe_acceptance": float(output_metrics["safe_output_acceptance"]) >= float(targets["output_safe_acceptance"]),
        "fault_injection": fault.get("status") == "passed" and float(fault.get("pass_rate", 0)) == 1.0,
    }
    eligible = all(checks.values())
    report = {
        "schema_version": "dep001d_development_assurance_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "eligible_to_freeze" if eligible else "not_eligible_to_freeze",
        "candidate_freeze_eligible": eligible,
        "case_n": len(rows),
        "integrated_metrics": metrics,
        "confidence_intervals": integrated["confidence_intervals"],
        "output_actionability_internal_test": output_metrics,
        "fault_injection": {
            "status": fault.get("status"),
            "pass_rate": fault.get("pass_rate"),
            "passed_n": fault.get("passed_n"),
            "total_n": fault.get("total_n"),
        },
        "checks": checks,
        "development_targets": targets,
        "training_status": training.get("status"),
        "dep001c_consumed_bank_evaluated": False,
        "blind_bank_evaluated": False,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "limitations": [
            "The corpus is synthetic and programmatically composed.",
            "Perfect separation can reflect grammar-family homogeneity.",
            "This is development evidence, not blind, external, or clinical validation.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected_json_object")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


__all__ = ["OUTPUT_PATH", "build_dep001d_development_assurance"]
