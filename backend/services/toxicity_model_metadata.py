from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.artifact_manifest import build_artifact_manifest
from backend.services.toxicity_shortcut_audit import DEFAULT_OUTPUT_PATH as TOXICITY_AUDIT_PATH
from backend.services.toxicity_shortcut_audit import load_toxicity_shortcut_audit


DEFAULT_MODEL_PATH = "Data/complete_synthetic_training/gradient_boosting_toxicity_risk_binary.joblib"
DEFAULT_METADATA_PATH = "Data/complete_synthetic_training/gradient_boosting_toxicity_risk_binary.metadata.json"


def build_toxicity_model_metadata(
    model_path: str = DEFAULT_MODEL_PATH,
    output_path: str = DEFAULT_METADATA_PATH,
) -> dict[str, Any]:
    audit = load_toxicity_shortcut_audit(TOXICITY_AUDIT_PATH)
    model_exists = Path(model_path).exists()
    metadata = {
        **build_artifact_manifest(dataset_paths={"toxicity_model": model_path, "shortcut_audit": TOXICITY_AUDIT_PATH}),
        "schema_version": "toxicity_model_metadata_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "needs_attention" if audit.get("status") == "needs_attention" else "documentation_only",
        "model_path": model_path,
        "model_exists": model_exists,
        "task": "toxicity_risk_binary",
        "label_source": "synthetic_generator_threshold_label",
        "known_shortcut_risk": {
            "status": audit.get("status", "missing"),
            "direct_rule_reconstruction": audit.get("rule_reconstruction", {}).get("direct_rule_reconstruction"),
            "rule_accuracy": audit.get("rule_reconstruction", {}).get("accuracy"),
            "rule_auroc": audit.get("rule_reconstruction", {}).get("auroc"),
        },
        "recommended_use": {
            "current": "review_flag_or_deterministic_monitoring_rule",
            "not_supported": [
                "learned clinical toxicity prediction",
                "patient-specific treatment adjustment",
                "clinical adverse-event grading without clinician review",
            ],
            "promotion_requirement": (
                "Replace the deterministic synthetic label with clinician-reviewed adverse-event labels, "
                "lagged pre-outcome features, calibrated uncertainty, and external validation."
            ),
        },
        "claim_boundary": (
            "This metadata documents a synthetic toxicity head whose label is largely reconstructable from "
            "generator rules. It is a transparency artifact, not evidence of clinical toxicity prediction."
        ),
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_toxicity_model_metadata(output_path: str = DEFAULT_METADATA_PATH) -> dict[str, Any]:
    path = Path(output_path)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "schema_version": "toxicity_model_metadata_v1",
        "status": "missing",
        "message": "Run scripts/run_toxicity_model_metadata.py to generate this artifact.",
    }


__all__ = [
    "build_toxicity_model_metadata",
    "load_toxicity_model_metadata",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_METADATA_PATH",
]
