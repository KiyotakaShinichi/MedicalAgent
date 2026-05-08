import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backend.services.complete_synthetic_training import CATEGORICAL_FEATURES, EXCLUDED_COLUMNS, NUMERIC_FEATURES


DEFAULT_TRAINING_ROWS_PATH = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
DEFAULT_OUTPUT_PATH = "Data/complete_synthetic_training/leakage_audit/temporal_leakage_audit.json"


FUTURE_OR_LABEL_TERMS = {
    "final",
    "outcome",
    "success",
    "maintenance",
    "assessment",
    "latent",
    "label",
    "target",
}


def run_temporal_leakage_audit(
    training_rows_path=DEFAULT_TRAINING_ROWS_PATH,
    output_path=DEFAULT_OUTPUT_PATH,
):
    rows = pd.read_csv(training_rows_path)
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    findings = []

    missing_features = [column for column in feature_columns if column not in rows.columns]
    findings.append(_finding(
        "feature_columns_present",
        "passed" if not missing_features else "failed",
        {"missing": missing_features},
        "The training feature contract must resolve to columns in temporal_ml_rows.csv.",
    ))

    suspicious_features = [
        column for column in feature_columns
        if any(term in column.lower() for term in FUTURE_OR_LABEL_TERMS)
    ]
    findings.append(_finding(
        "no_future_or_label_named_features",
        "passed" if not suspicious_features else "failed",
        {"suspicious_features": suspicious_features},
        "Feature names should not include final outcomes, labels, or assessment-only variables.",
    ))

    excluded_present = sorted(set(EXCLUDED_COLUMNS).intersection(rows.columns))
    features_using_excluded = sorted(set(feature_columns).intersection(EXCLUDED_COLUMNS))
    findings.append(_finding(
        "excluded_columns_not_used_as_features",
        "passed" if not features_using_excluded else "failed",
        {
            "excluded_columns_available_in_dataset": excluded_present,
            "excluded_columns_used_as_features": features_using_excluded,
        },
        "Outcome and bookkeeping columns may exist in the dataset but must not be used by the model.",
    ))

    findings.extend(_timeline_findings(rows))
    findings.extend(_current_cycle_feature_findings(rows))

    status = "failed" if any(item["status"] == "failed" for item in findings) else "passed"
    payload = {
        "schema_version": "temporal_leakage_audit_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "training_rows_path": training_rows_path,
        "feature_columns": feature_columns,
        "findings": findings,
        "interpretation": (
            "This audit verifies that model features are current-cycle or historical monitoring fields. "
            "Final outcome labels are present for supervised learning but excluded from model inputs."
        ),
        "claim_boundary": "A passed audit reduces obvious synthetic timeline leakage risk; it is not proof against every clinical data leakage mode.",
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _timeline_findings(rows):
    findings = []
    if {"patient_id", "cycle"}.issubset(rows.columns):
        ordered = rows.sort_values(["patient_id", "cycle"])
        duplicate_cycles = int(ordered.duplicated(["patient_id", "cycle"]).sum())
        non_monotonic_patients = []
        for patient_id, group in ordered.groupby("patient_id"):
            cycles = pd.to_numeric(group["cycle"], errors="coerce").dropna().tolist()
            if cycles != sorted(cycles):
                non_monotonic_patients.append(patient_id)
        findings.append(_finding(
            "one_row_per_patient_cycle",
            "passed" if duplicate_cycles == 0 else "failed",
            {"duplicate_patient_cycle_rows": duplicate_cycles},
            "Duplicate patient-cycle rows can leak repeated labels or overweight specific visits.",
        ))
        findings.append(_finding(
            "cycle_order_monotonic_by_patient",
            "passed" if not non_monotonic_patients else "failed",
            {"example_non_monotonic_patients": non_monotonic_patients[:10]},
            "Temporal rows should be ordered by observed treatment cycle.",
        ))
    if {"patient_id", "cycle", "treatment_date"}.issubset(rows.columns):
        dated = rows.copy()
        dated["treatment_date"] = pd.to_datetime(dated["treatment_date"], errors="coerce")
        inconsistent = []
        for patient_id, group in dated.sort_values(["patient_id", "cycle"]).groupby("patient_id"):
            dates = group["treatment_date"].dropna().tolist()
            if dates != sorted(dates):
                inconsistent.append(patient_id)
        findings.append(_finding(
            "treatment_dates_follow_cycle_order",
            "passed" if not inconsistent else "failed",
            {"example_inconsistent_patients": inconsistent[:10]},
            "Later cycles should not have earlier treatment dates.",
        ))
    return findings


def _current_cycle_feature_findings(rows):
    findings = []
    if {"mri_percent_change_from_baseline", "response_score_percent"}.issubset(rows.columns):
        diff = (
            pd.to_numeric(rows["response_score_percent"], errors="coerce")
            + pd.to_numeric(rows["mri_percent_change_from_baseline"], errors="coerce")
        ).abs()
        findings.append(_finding(
            "response_regression_label_is_mri_transform",
            "passed" if float(diff.dropna().max() or 0) <= 0.05 else "failed",
            {"max_absolute_transform_gap": round(float(diff.dropna().max() or 0), 4)},
            "The continuous response label is a transparent transform of the observed MRI percent-change signal.",
        ))
    return findings


def _finding(name, status, evidence, meaning):
    return {
        "name": name,
        "status": status,
        "evidence": evidence,
        "meaning": meaning,
    }
