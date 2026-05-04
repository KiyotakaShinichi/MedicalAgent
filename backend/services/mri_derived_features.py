import math

import pandas as pd


def build_mri_derived_feature_summary(evaluation_frame=None, mri_reports=None):
    report_summary = build_mri_report_feature_pipeline(mri_reports)
    if evaluation_frame is None or evaluation_frame.empty or "latest_mri_percent_change" not in evaluation_frame.columns:
        return {
            "status": report_summary.get("status", "unavailable"),
            "purpose": "MRI-derived features are expected, but no evaluation feature frame was available.",
            "features": [],
            "report_pipeline": report_summary,
        }

    changes = evaluation_frame["latest_mri_percent_change"].dropna().astype(float)
    sizes = (
        evaluation_frame["latest_mri_tumor_size_cm"].dropna().astype(float)
        if "latest_mri_tumor_size_cm" in evaluation_frame.columns
        else pd.Series(dtype=float)
    )
    if changes.empty:
        return {
            "status": "unavailable",
            "purpose": "MRI-derived features exist in schema, but values are missing for this evaluation split.",
            "features": [],
            "report_pipeline": report_summary,
        }

    return {
        "status": "acceptable",
        "purpose": "Summarizes MRI-derived numeric trend features used by the longitudinal simulator; this is not raw MRI interpretation.",
        "features": [
            {
                "name": "latest_mri_percent_change",
                "meaning": "Percent tumor-size change from baseline MRI-derived measurement.",
                "mean": _round(changes.mean()),
                "min": _round(changes.min()),
                "max": _round(changes.max()),
            },
            {
                "name": "latest_mri_tumor_size_cm",
                "meaning": "Latest tumor-size measurement in centimeters when available.",
                "mean": _round(sizes.mean()) if not sizes.empty else None,
                "min": _round(sizes.min()) if not sizes.empty else None,
                "max": _round(sizes.max()) if not sizes.empty else None,
            },
        ],
        "response_trend_buckets": _trend_buckets(changes),
        "report_pipeline": report_summary,
        "safety_note": "Raw DICOM/NIfTI computer vision remains a planned integration path. Current longitudinal models use MRI-derived tabular features.",
    }


def build_mri_report_feature_pipeline(mri_reports):
    if mri_reports is None or mri_reports.empty or "patient_id" not in mri_reports.columns:
        return {
            "status": "unavailable",
            "purpose": "No MRI report table was available for derived-feature inventory.",
            "steps": [],
        }

    reports = mri_reports.copy()
    reports["date"] = pd.to_datetime(reports.get("date"), errors="coerce")
    sort_columns = [column for column in ["patient_id", "date", "cycle"] if column in reports.columns]
    reports = reports.sort_values(sort_columns)
    patient_count = int(reports["patient_id"].nunique())
    baseline = (
        reports[reports.get("timepoint", "").astype(str).str.lower().eq("baseline")]
        if "timepoint" in reports.columns
        else pd.DataFrame()
    )
    followup = reports[~reports.index.isin(baseline.index)] if not baseline.empty else reports

    patient_latest = reports.groupby("patient_id", as_index=False).tail(1)
    latest_changes = (
        patient_latest["percent_change_from_baseline"].dropna().astype(float)
        if "percent_change_from_baseline" in patient_latest.columns
        else pd.Series(dtype=float)
    )
    size_values = (
        reports["tumor_size_cm"].dropna().astype(float)
        if "tumor_size_cm" in reports.columns
        else pd.Series(dtype=float)
    )
    coverage = float(len(latest_changes) / patient_count) if patient_count else 0.0

    return {
        "status": _coverage_status(coverage),
        "purpose": "Inventory of the MRI-derived feature pipeline from synthetic MRI measurements.",
        "patients_with_mri": patient_count,
        "measurement_rows": int(len(reports)),
        "patients_with_baseline": int(baseline["patient_id"].nunique()) if not baseline.empty else 0,
        "patients_with_followup": int(followup["patient_id"].nunique()) if not followup.empty else 0,
        "latest_change_coverage": _round(coverage),
        "tumor_size_cm_mean": _round(size_values.mean()) if not size_values.empty else None,
        "latest_percent_change_mean": _round(latest_changes.mean()) if not latest_changes.empty else None,
        "response_trend_buckets": _trend_buckets(latest_changes),
        "steps": [
            "Read one synthetic MRI measurement row per patient baseline and treatment-cycle follow-up.",
            "Sort measurements by patient and date.",
            "Use baseline tumor size as the reference measurement.",
            "Compute latest tumor size and percent change from baseline.",
            "Bucket MRI-derived trend as strong decrease, partial decrease, stable/weak decrease, or increase.",
            "Join MRI-derived trend features into the longitudinal treatment model table.",
        ],
    }


def _trend_buckets(changes):
    if changes is None or changes.empty:
        return {
            "strong_decrease": 0,
            "partial_decrease": 0,
            "stable_or_weak_decrease": 0,
            "increase": 0,
        }
    return {
        "strong_decrease": int((changes <= -50).sum()),
        "partial_decrease": int(((changes > -50) & (changes <= -20)).sum()),
        "stable_or_weak_decrease": int(((changes > -20) & (changes <= 10)).sum()),
        "increase": int((changes > 10).sum()),
    }


def _coverage_status(value):
    if value is None:
        return "unavailable"
    if value < 0.70:
        return "failed"
    if value < 0.85:
        return "unideal"
    if value < 0.95:
        return "acceptable"
    return "passed"


def _round(value, digits=3):
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return round(numeric, digits)

