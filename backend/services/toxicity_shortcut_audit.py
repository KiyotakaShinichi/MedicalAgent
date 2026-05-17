from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from backend.services.artifact_manifest import build_artifact_manifest
from backend.services.biomarker_feature_benchmark import DEFAULT_SOURCE_CSV


DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_toxicity_shortcut_audit.json"
TOXICITY_LABEL = "toxicity_risk_binary"


def run_toxicity_shortcut_audit(
    source_csv: str = DEFAULT_SOURCE_CSV,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    rows = pd.read_csv(source_csv)
    required = {
        TOXICITY_LABEL,
        "nadir_anc",
        "nadir_hemoglobin",
        "nadir_platelets",
        "max_symptom_severity",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Missing required toxicity audit columns: {missing}")

    y = rows[TOXICITY_LABEL].astype(int).to_numpy()
    rule_score = _rule_score(rows)
    rule_pred = (rule_score >= 1.0).astype(int)
    agreement = float(accuracy_score(y, rule_pred))
    auc = _safe_auc(y, rule_score)
    component_rates = {
        "nadir_anc_lt_1_1": float((pd.to_numeric(rows["nadir_anc"], errors="coerce") < 1.1).mean()),
        "nadir_hgb_lt_8_3": float((pd.to_numeric(rows["nadir_hemoglobin"], errors="coerce") < 8.3).mean()),
        "nadir_platelets_lt_60": float((pd.to_numeric(rows["nadir_platelets"], errors="coerce") < 60).mean()),
        "max_symptom_ge_8": float((pd.to_numeric(rows["max_symptom_severity"], errors="coerce") >= 8).mean()),
    }
    direct_rule_reconstruction = agreement >= 0.98
    status = "needs_attention" if direct_rule_reconstruction or (auc is not None and auc >= 0.98) else "acceptable"
    report = {
        **build_artifact_manifest(dataset_paths={"toxicity_source_rows": source_csv}),
        "schema_version": "toxicity_shortcut_audit_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "claim_boundary": (
            "Synthetic toxicity shortcut audit. A high score means the toxicity label is reconstructable from "
            "generator threshold rules; it is a warning about simulator determinism, not evidence of clinical "
            "toxicity prediction."
        ),
        "source_csv": source_csv,
        "rows": int(len(rows)),
        "positive_label_rate": float(np.mean(y)),
        "rule_reconstruction": {
            "rule": "nadir_anc < 1.1 OR nadir_hemoglobin < 8.3 OR nadir_platelets < 60 OR max_symptom_severity >= 8",
            "accuracy": round(agreement, 4),
            "auroc": auc,
            "direct_rule_reconstruction": direct_rule_reconstruction,
            "component_positive_rates": {k: round(v, 4) for k, v in component_rates.items()},
        },
        "recommendation": {
            "use": "deterministic_monitoring_rule_or_review_flag",
            "do_not_claim": "Do not present AUC=1.0 toxicity as learned clinical prediction.",
            "next_step": (
                "If toxicity prediction remains a project goal, create a softer label with clinician-reviewed "
                "adverse-event grades, lagged features, noise, and external validation."
            ),
        },
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_toxicity_shortcut_audit(output_path: str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    path = Path(output_path)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "schema_version": "toxicity_shortcut_audit_v1",
        "status": "missing",
        "message": "Run scripts/run_toxicity_shortcut_audit.py to generate this artifact.",
    }


def _rule_score(rows: pd.DataFrame) -> np.ndarray:
    return (
        (pd.to_numeric(rows["nadir_anc"], errors="coerce") < 1.1).astype(int)
        + (pd.to_numeric(rows["nadir_hemoglobin"], errors="coerce") < 8.3).astype(int)
        + (pd.to_numeric(rows["nadir_platelets"], errors="coerce") < 60).astype(int)
        + (pd.to_numeric(rows["max_symptom_severity"], errors="coerce") >= 8).astype(int)
    ).to_numpy()


def _safe_auc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return round(float(roc_auc_score(y_true, score)), 4)


__all__ = ["run_toxicity_shortcut_audit", "load_toxicity_shortcut_audit", "DEFAULT_OUTPUT_PATH"]
