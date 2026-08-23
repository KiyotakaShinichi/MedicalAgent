"""ORM persistence for generated synthetic journey tables."""

from backend.models import (
    BreastCancerProfile,
    ClinicalIntervention,
    ImagingReport,
    LabResult,
    MedicationLog,
    Patient,
    SymptomReport,
    Treatment,
    TreatmentOutcome,
)


def _write_journey_to_db(db, journey):
    patient = journey["patients"][0]
    diagnosis = journey["diagnoses"][0]
    db.add(
        Patient(
            id=patient["patient_id"],
            name=patient["name"],
            diagnosis=patient["diagnosis"],
        )
    )
    db.add(
        BreastCancerProfile(
            patient_id=patient["patient_id"],
            cancer_stage=diagnosis["stage"],
            er_status=diagnosis["er_status"],
            pr_status=diagnosis["pr_status"],
            her2_status=diagnosis["her2_status"],
            molecular_subtype=diagnosis["molecular_subtype"],
            treatment_intent=diagnosis["treatment_intent"],
            menopausal_status=diagnosis["menopausal_status"],
        )
    )
    for row in journey["treatment_sessions"]:
        db.add(
            Treatment(
                patient_id=row["patient_id"],
                date=row["actual_date"],
                cycle=row["cycle"],
                drug=row["regimen"],
            )
        )
    for row in journey["labs"]:
        db.add(
            LabResult(
                patient_id=row["patient_id"],
                date=row["date"],
                wbc=row["wbc"],
                hemoglobin=row["hemoglobin"],
                platelets=row["platelets"],
                source=row["source"],
                source_note=(
                    f"{row['lab_timepoint']}: ANC {row['anc']}, RBC {row['rbc']}. "
                    f"{row['note']}"
                ),
            )
        )
    for row in journey["medications"]:
        db.add(
            MedicationLog(
                patient_id=row["patient_id"],
                date=row["date"],
                medication=row["medication"],
                dose=row["dose"],
                frequency=row["frequency"],
                notes=row["notes"],
                source=row["source"],
            )
        )
    for row in journey["symptoms"]:
        db.add(
            SymptomReport(
                patient_id=row["patient_id"],
                date=row["date"],
                symptom=row["symptom"],
                severity=row["severity"],
                notes=row["notes"],
            )
        )
    for row in journey["mri_reports"]:
        db.add(
            ImagingReport(
                patient_id=row["patient_id"],
                date=row["date"],
                modality=row["modality"],
                report_type=f"Synthetic {row['timepoint']} {row['modality']}",
                body_site=row.get("body_site") or "Breast",
                findings=(
                    f"Synthetic report: {row.get('breast_side') or ''} "
                    f"{row.get('location') or ''} "
                    f"{'measuring ' + str(row['tumor_size_cm']) + ' cm. ' if row.get('tumor_size_cm') is not None else ''}"
                    f"{row['response_text']}."
                ),
                impression=(
                    f"Synthetic {row['modality']} report for software testing only."
                ),
            )
        )
    for row in journey["interventions"]:
        db.add(
            ClinicalIntervention(
                patient_id=row["patient_id"],
                date=row["date"],
                intervention_type=row["intervention_type"],
                reason=row["reason"],
                medication_or_product=row["medication_or_product"],
                dose=row["dose"],
                notes=row["notes"],
                source=row["source"],
            )
        )
    outcome = journey["outcomes"][0]
    db.add(
        TreatmentOutcome(
            patient_id=outcome["patient_id"],
            assessment_date=outcome["assessment_date"],
            response_category=outcome["response_category"],
            cancer_status=outcome["cancer_status"],
            maintenance_plan=outcome["maintenance_plan"],
            recurrence_risk_band=outcome["recurrence_risk_band"],
            notes=outcome["notes"],
            source=outcome["source"],
        )
    )
