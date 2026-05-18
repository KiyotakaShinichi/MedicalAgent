from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss

from backend.services.artifact_manifest import build_artifact_manifest
from backend.services.biomarker_feature_benchmark import DEFAULT_SOURCE_CSV
from backend.services.hybrid_prediction import predict_response_score_with_abstention, predict_toxicity_with_abstention
from backend.services.predict_with_abstention import predict_with_abstention


DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_hybrid_subgroup_metrics.json"


def run_hybrid_subgroup_metrics(
    source_csv: str = DEFAULT_SOURCE_CSV,
    output_path: str = DEFAULT_OUTPUT_PATH,
    max_rows: int = 900,
) -> dict[str, Any]:
    rows = pd.read_csv(source_csv).tail(max_rows).copy()
    scored = []
    for _, row in rows.iterrows():
        item = row.to_dict()
        cls = predict_with_abstention(item)
        reg = predict_response_score_with_abstention(item)
        tox = predict_toxicity_with_abstention(item)
        scored.append({
            "stage": item.get("stage"),
            "molecular_subtype": item.get("molecular_subtype"),
            "classification_covered": cls.probability is not None,
            "classification_prob": cls.probability,
            "classification_correct": _cls_correct(cls.probability, item.get("treatment_success_binary")),
            "classification_brier_y": item.get("treatment_success_binary"),
            "regression_covered": reg.response_score is not None,
            "regression_abs_error": _abs_error(reg.response_score, item.get("response_score_percent")),
            "toxicity_covered": tox.probability is not None,
            "toxicity_prob": tox.probability,
            "toxicity_correct": _cls_correct(tox.probability, item.get("toxicity_risk_binary")),
            "toxicity_brier_y": item.get("toxicity_risk_binary"),
        })
    frame = pd.DataFrame(scored)
    report = {
        **build_artifact_manifest(dataset_paths={"source_csv": source_csv}),
        "schema_version": "hybrid_subgroup_metrics_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong" if len(frame) else "missing",
        "claim_boundary": (
            "Synthetic subgroup metric table for engineering fairness/calibration inspection. "
            "Subgroups are simulator-defined and do not establish real-world equity or clinical validity."
        ),
        "overall": _metrics(frame),
        "by_molecular_subtype": _group_metrics(frame, "molecular_subtype"),
        "by_stage": _group_metrics(frame, "stage"),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_hybrid_subgroup_metrics(path: str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"schema_version": "hybrid_subgroup_metrics_v1", "status": "missing"}


def _group_metrics(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    return [{"group": str(k), **_metrics(g)} for k, g in frame.groupby(column, dropna=False)]


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    cls = frame[frame["classification_covered"]]
    tox = frame[frame["toxicity_covered"]]
    reg = frame[frame["regression_covered"]]
    return {
        "n": int(len(frame)),
        "classification_coverage": round(float(frame["classification_covered"].mean()), 4) if len(frame) else None,
        "classification_accuracy": _mean_bool(cls["classification_correct"]) if len(cls) else None,
        "classification_brier": _brier(cls, "classification_brier_y", "classification_prob"),
        "regression_coverage": round(float(frame["regression_covered"].mean()), 4) if len(frame) else None,
        "regression_mae": round(float(reg["regression_abs_error"].mean()), 4) if len(reg) else None,
        "toxicity_coverage": round(float(frame["toxicity_covered"].mean()), 4) if len(frame) else None,
        "toxicity_accuracy": _mean_bool(tox["toxicity_correct"]) if len(tox) else None,
        "toxicity_brier": _brier(tox, "toxicity_brier_y", "toxicity_prob"),
    }


def _cls_correct(prob, target) -> bool | None:
    if prob is None or pd.isna(prob) or pd.isna(target):
        return None
    return bool((float(prob) >= 0.5) == bool(int(target)))


def _abs_error(score, target_percent) -> float | None:
    if score is None or pd.isna(score) or pd.isna(target_percent):
        return None
    return abs(float(score) - float(target_percent) / 100.0)


def _mean_bool(series: pd.Series) -> float | None:
    vals = [v for v in series.tolist() if v is not None and not pd.isna(v)]
    return round(float(np.mean(vals)), 4) if vals else None


def _brier(frame: pd.DataFrame, target_col: str, prob_col: str) -> float | None:
    clean = frame[[target_col, prob_col]].dropna()
    if clean.empty:
        return None
    return round(float(brier_score_loss(clean[target_col].astype(int), clean[prob_col].astype(float))), 4)


__all__ = ["run_hybrid_subgroup_metrics", "load_hybrid_subgroup_metrics", "DEFAULT_OUTPUT_PATH"]
