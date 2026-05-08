import json
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

from backend.models import ImagingReport, LabResult, Patient, SymptomReport, Treatment, TreatmentOutcome


DEFAULT_COHERENCE_AUDIT_PATH = "Data/data_quality/patient_data_coherence.json"


def audit_patient_data_coherence(db, output_path=DEFAULT_COHERENCE_AUDIT_PATH, include_examples=True):
    patients = db.query(Patient).order_by(Patient.id).all()
    rows = [_audit_patient(db, patient, include_examples=include_examples) for patient in patients]
    issue_counts = Counter(issue["code"] for row in rows for issue in row["issues"])
    high_priority = [
        row for row in rows
        if any(issue["severity"] in {"error", "warning"} for issue in row["issues"])
    ]
    payload = {
        "schema_version": "patient_data_coherence_v1",
        "patient_count": len(rows),
        "status": "passed" if not high_priority else "needs_review",
        "issue_counts": dict(sorted(issue_counts.items())),
        "patients_with_warnings_or_errors": len(high_priority),
        "patients": rows,
        "interpretation": (
            "Imported imaging-only cohorts may intentionally have MRI metadata without CBC/treatment cycles. "
            "Synthetic monitoring-demo cohorts are expected to have synchronized treatments, CBC windows, and MRI timelines."
        ),
    }
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def _audit_patient(db, patient, include_examples):
    treatments = db.query(Treatment).filter(Treatment.patient_id == patient.id).order_by(Treatment.date, Treatment.cycle).all()
    labs = db.query(LabResult).filter(LabResult.patient_id == patient.id).order_by(LabResult.date).all()
    imaging = db.query(ImagingReport).filter(ImagingReport.patient_id == patient.id).order_by(ImagingReport.date).all()
    symptoms = db.query(SymptomReport).filter(SymptomReport.patient_id == patient.id).order_by(SymptomReport.date).all()
    outcomes = db.query(TreatmentOutcome).filter(TreatmentOutcome.patient_id == patient.id).all()

    cohort = _cohort_type(patient.id)
    issues = []
    same_date_groups = _same_date_treatments(treatments)
    if same_date_groups:
        issues.append(_issue(
            "same_date_treatment_components",
            "warning",
            "Multiple treatment rows share one date; they should usually be one combined regimen/cycle.",
            same_date_groups if include_examples else None,
        ))

    duplicate_cycles = [
        cycle for cycle, count in Counter(row.cycle for row in treatments).items()
        if count > 1
    ]
    if duplicate_cycles:
        issues.append(_issue(
            "duplicate_treatment_cycle_numbers",
            "error",
            "Treatment cycle numbers repeat for this patient.",
            sorted(duplicate_cycles)[:8] if include_examples else None,
        ))

    duplicate_imaging = _duplicate_imaging_rows(imaging)
    if duplicate_imaging:
        issues.append(_issue(
            "duplicate_imaging_rows",
            "warning",
            "Exact duplicate imaging report rows were found.",
            duplicate_imaging[:8] if include_examples else None,
        ))

    if treatments and labs:
        missing_windows = _missing_lab_windows(treatments, labs)
        if missing_windows:
            issues.append(_issue(
                "treatment_cycles_missing_nearby_cbc",
                "warning",
                "One or more treatment cycles lack nearby CBC values for monitoring context.",
                missing_windows[:8] if include_examples else None,
            ))
    elif treatments and not labs:
        issues.append(_issue("treatments_without_cbc", "warning", "Treatment cycles exist but no CBC rows are present."))
    elif labs and not treatments:
        severity = "info" if cohort in {"patient_entered_or_partial", "external_imaging_only"} else "warning"
        issues.append(_issue("cbc_without_treatments", severity, "CBC rows exist but no treatment cycles are present."))

    if _is_monitoring_demo(cohort):
        _monitoring_demo_expectations(issues, cohort, treatments, labs, imaging, outcomes)
    elif imaging and not treatments and not labs:
        issues.append(_issue(
            "imaging_only_dataset",
            "info",
            "This appears to be an imported imaging-only cohort, not a full longitudinal monitoring journey.",
        ))

    if treatments and imaging:
        baseline_mri = imaging[0]
        first_treatment = treatments[0]
        if baseline_mri.date > first_treatment.date + timedelta(days=7):
            issues.append(_issue(
                "mri_starts_after_treatment",
                "warning",
                "First MRI occurs after treatment start; timeline may not have a baseline imaging signal.",
                {"first_mri": baseline_mri.date, "first_treatment": first_treatment.date} if include_examples else None,
            ))

    return {
        "patient_id": patient.id,
        "name": patient.name,
        "cohort_type": cohort,
        "counts": {
            "treatments": len(treatments),
            "labs": len(labs),
            "imaging_reports": len(imaging),
            "symptoms": len(symptoms),
            "outcomes": len(outcomes),
        },
        "timeline_span": _timeline_span(treatments, labs, imaging, symptoms),
        "status": "needs_review" if any(issue["severity"] in {"error", "warning"} for issue in issues) else "passed",
        "issues": issues,
    }


def _cohort_type(patient_id):
    if patient_id == "P001":
        return "curated_demo_patient"
    if patient_id.startswith("COMP"):
        return "complete_synthetic_monitoring"
    if patient_id.startswith("SYN-TEMP-BRCA-"):
        return "temporal_synthetic_monitoring"
    if patient_id.startswith("SYN-BRCA-"):
        return "simple_synthetic_monitoring"
    if patient_id.startswith("QIN-BREAST-02-"):
        return "qin_external_with_synthetic_cbc"
    if patient_id.startswith("ISPY1_"):
        return "external_imaging_only"
    return "patient_entered_or_partial"


def _is_monitoring_demo(cohort):
    return cohort in {
        "curated_demo_patient",
        "complete_synthetic_monitoring",
        "temporal_synthetic_monitoring",
        "simple_synthetic_monitoring",
        "qin_external_with_synthetic_cbc",
    }


def _monitoring_demo_expectations(issues, cohort, treatments, labs, imaging, outcomes):
    if len(treatments) >= 1 and len(labs) < (1 + 2 * len(treatments)):
        issues.append(_issue(
            "low_cbc_density_for_cycles",
            "warning",
            "Synthetic monitoring journey has fewer CBC rows than expected for baseline/nadir/recovery style tracking.",
            {"treatments": len(treatments), "labs": len(labs)},
        ))
    if cohort != "qin_external_with_synthetic_cbc" and len(treatments) >= 4 and len(imaging) < 2:
        issues.append(_issue(
            "low_mri_density_for_monitoring",
            "warning",
            "Synthetic monitoring journey has treatment cycles but only one MRI/imaging report.",
            {"treatments": len(treatments), "imaging_reports": len(imaging)},
        ))
    if len(treatments) >= 6 and not outcomes:
        issues.append(_issue(
            "missing_synthetic_outcome",
            "info",
            "Full synthetic journey has no outcome label; this can be okay for partial demos but not model training.",
        ))


def _missing_lab_windows(treatments, labs):
    missing = []
    lab_dates = [row.date for row in labs]
    for treatment in treatments:
        has_nearby = any(
            treatment.date - timedelta(days=7) <= lab_date <= treatment.date + timedelta(days=21)
            for lab_date in lab_dates
        )
        if not has_nearby:
            missing.append({"cycle": treatment.cycle, "date": treatment.date})
    return missing


def _same_date_treatments(treatments):
    groups = defaultdict(list)
    for row in treatments:
        groups[row.date].append({"cycle": row.cycle, "drug": row.drug})
    return [
        {"date": date_value, "rows": rows}
        for date_value, rows in groups.items()
        if len(rows) > 1
    ]


def _duplicate_imaging_rows(imaging):
    counter = Counter(
        (
            row.date,
            row.modality,
            row.report_type,
            row.findings,
            row.impression,
        )
        for row in imaging
    )
    return [
        {"date": key[0], "modality": key[1], "report_type": key[2], "count": count}
        for key, count in counter.items()
        if count > 1
    ]


def _timeline_span(*row_groups):
    dates = [row.date for rows in row_groups for row in rows if row.date]
    if not dates:
        return None
    return {"start": min(dates), "end": max(dates)}


def _issue(code, severity, message, examples=None):
    issue = {"code": code, "severity": severity, "message": message}
    if examples is not None:
        issue["examples"] = examples
    return issue
