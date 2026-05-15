"""
TCGA-BRCA external evaluation scaffold.

TCGA-BRCA is useful for real clinical distribution checks, subtype/stage
coverage, and survival/outcome context. It is not a drop-in match for this
project's synthetic longitudinal CBC + treatment-monitoring target, so this
module is intentionally conservative:

- It can download a public clinical snapshot from the NCI GDC API.
- It writes an applicability report showing which OncoTrack features/labels
  are available, missing, or require mapping.
- If a mapped CSV with actual_label and predicted_probability is supplied,
  it computes external AUROC/Brier/confusion metrics.

This avoids the common portfolio mistake of calling any public clinical cohort
"external validation" when the target label does not actually match.
"""

from __future__ import annotations

import csv
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


DEFAULT_OUTPUT_DIR = Path("Data/external_validation/tcga_brca")
GDC_CASES_URL = "https://api.gdc.cancer.gov/cases"

REQUIRED_ONCOTRACK_FEATURES = {
    "age",
    "stage",
    "molecular_subtype",
    "treatment_cycle",
    "days_since_treatment",
    "wbc",
    "hemoglobin",
    "platelets",
    "symptom_severity",
    "imaging_response_signal",
}

TARGETS_REQUIRED_FOR_MODEL_EVAL = {"actual_label", "predicted_probability"}


def build_tcga_brca_external_eval(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    mapped_predictions_csv: str | Path | None = None,
    download: bool = True,
    size: int = 2000,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    clinical_csv = output / "tcga_brca_clinical_snapshot.csv"

    download_status = "skipped"
    download_error = None
    if download:
        try:
            rows = fetch_tcga_brca_clinical_rows(size=size)
            _write_rows(clinical_csv, rows)
            download_status = "downloaded"
        except Exception as exc:  # network/API failures should not break local gates
            download_status = "failed"
            download_error = str(exc)[:300]

    clinical = pd.read_csv(clinical_csv) if clinical_csv.exists() else pd.DataFrame()
    applicability = _applicability_report(clinical)
    mapped_metrics = _evaluate_mapped_predictions(mapped_predictions_csv) if mapped_predictions_csv else None

    if mapped_metrics:
        status = mapped_metrics["status"]
        message = "Mapped TCGA-BRCA prediction file evaluated."
    elif not clinical.empty:
        status = "external_distribution_available_model_metrics_not_computed"
        message = (
            "TCGA-BRCA clinical rows were loaded, but the public snapshot does not include "
            "OncoTrack's longitudinal CBC/treatment-response target. Use it for distribution "
            "checks unless a mapped prediction CSV is provided."
        )
    else:
        status = "unavailable"
        message = "TCGA-BRCA clinical snapshot is unavailable in this environment."

    report = {
        "schema_version": "tcga_brca_external_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "TCGA-BRCA public clinical snapshot via NCI GDC",
        "status": status,
        "message": message,
        "rows": int(len(clinical)),
        "download": {
            "status": download_status,
            "error": download_error,
            "source_url": GDC_CASES_URL,
        },
        "applicability": applicability,
        "mapped_prediction_metrics": mapped_metrics,
        "files": {
            "clinical_snapshot_csv": str(clinical_csv),
            "report_json": str(output / "tcga_brca_external_eval.json"),
        },
        "claim_boundary": (
            "TCGA-BRCA is a real external public cohort, but this report is not clinical "
            "validation of the synthetic monitoring model unless mapped labels/predictions "
            "with the same target are supplied."
        ),
    }
    Path(report["files"]["report_json"]).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def fetch_tcga_brca_clinical_rows(*, size: int = 2000) -> list[dict[str, Any]]:
    fields = [
        "case_id",
        "submitter_id",
        "demographic.age_at_index",
        "demographic.gender",
        "demographic.race",
        "demographic.ethnicity",
        "diagnoses.primary_diagnosis",
        "diagnoses.tumor_stage",
        "diagnoses.vital_status",
        "diagnoses.days_to_death",
        "diagnoses.days_to_last_follow_up",
        "diagnoses.treatments.treatment_type",
        "diagnoses.treatments.treatment_or_therapy",
    ]
    filters = {
        "op": "in",
        "content": {
            "field": "project.project_id",
            "value": ["TCGA-BRCA"],
        },
    }
    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": str(size),
    }
    url = f"{GDC_CASES_URL}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    hits = payload.get("data", {}).get("hits", [])
    return [_flatten_case(hit) for hit in hits]


def _flatten_case(hit: dict[str, Any]) -> dict[str, Any]:
    diagnoses = hit.get("diagnoses") or [{}]
    diagnosis = diagnoses[0] if diagnoses else {}
    demographic = hit.get("demographic") or {}
    treatments = diagnosis.get("treatments") or []
    treatment_types = sorted({str(t.get("treatment_type")) for t in treatments if t.get("treatment_type")})
    return {
        "case_id": hit.get("case_id"),
        "submitter_id": hit.get("submitter_id"),
        "age_at_index": demographic.get("age_at_index"),
        "gender": demographic.get("gender"),
        "race": demographic.get("race"),
        "ethnicity": demographic.get("ethnicity"),
        "primary_diagnosis": diagnosis.get("primary_diagnosis"),
        "tumor_stage": diagnosis.get("tumor_stage"),
        "vital_status": diagnosis.get("vital_status"),
        "days_to_death": diagnosis.get("days_to_death"),
        "days_to_last_follow_up": diagnosis.get("days_to_last_follow_up"),
        "treatment_types": "; ".join(treatment_types),
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _applicability_report(clinical: pd.DataFrame) -> dict[str, Any]:
    available = set()
    if "age_at_index" in clinical.columns:
        available.add("age")
    if "tumor_stage" in clinical.columns:
        available.add("stage")
    missing = sorted(REQUIRED_ONCOTRACK_FEATURES - available)
    return {
        "status": "partial_overlap" if not clinical.empty else "not_available",
        "available_feature_families": sorted(available),
        "missing_feature_families": missing,
        "target_label_available": False,
        "target_label_note": (
            "The public clinical snapshot does not provide the same longitudinal "
            "treatment-success/CBC monitoring label used by the synthetic model."
        ),
        "recommended_use": [
            "External distribution sanity check",
            "Stage/age/demographic coverage comparison",
            "Future mapped-label evaluation if a compatible target is curated",
        ],
    }


def _evaluate_mapped_predictions(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    frame = pd.read_csv(path)
    missing = TARGETS_REQUIRED_FOR_MODEL_EVAL - set(frame.columns)
    if missing:
        return {
            "status": "not_computed",
            "reason": f"mapped prediction CSV missing columns: {sorted(missing)}",
        }
    labels = frame["actual_label"].astype(float).astype(int).to_numpy()
    probabilities = frame["predicted_probability"].astype(float).to_numpy()
    predicted = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    has_two_classes = len(set(labels.tolist())) > 1
    return {
        "status": "computed" if has_two_classes else "not_computed_single_class",
        "rows": int(len(frame)),
        "accuracy": round(float(accuracy_score(labels, predicted)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(labels, predicted)), 4),
        "precision": round(float(precision_score(labels, predicted, zero_division=0)), 4),
        "sensitivity": round(float(recall_score(labels, predicted, zero_division=0)), 4),
        "specificity": round(float(tn / max(tn + fp, 1)), 4),
        "brier_score": round(float(brier_score_loss(labels, probabilities)), 4),
        "roc_auc": round(float(roc_auc_score(labels, probabilities)), 4) if has_two_classes else None,
        "average_precision": round(float(average_precision_score(labels, probabilities)), 4) if has_two_classes else None,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }
