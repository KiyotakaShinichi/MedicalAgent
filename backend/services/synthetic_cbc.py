from datetime import timedelta

from backend.models import LabResult, Patient


def generate_synthetic_cbc_for_qin_patients(db):
    patients = (
        db.query(Patient)
        .filter(Patient.id.like("QIN-BREAST-02-%"))
        .all()
    )

    created = 0
    skipped_patients = 0

    for patient in patients:
        existing_synthetic = (
            db.query(LabResult)
            .filter(
                LabResult.patient_id == patient.id,
                LabResult.source == "synthetic_qin_demo",
            )
            .first()
        )
        if existing_synthetic:
            skipped_patients += 1
            continue

        treatments = sorted(patient_treatments(db, patient.id), key=lambda item: item.date)
        if not treatments:
            skipped_patients += 1
            continue

        baseline_date = treatments[0].date - timedelta(days=3)
        db.add(LabResult(
            patient_id=patient.id,
            date=baseline_date,
            wbc=6.2,
            hemoglobin=12.8,
            platelets=245,
            source="synthetic_qin_demo",
            source_note="Synthetic CBC for demo only. QIN-BREAST-02 does not provide serial CBC values.",
        ))
        created += 1

        for index, treatment in enumerate(treatments, start=1):
            nadir_date = treatment.date + timedelta(days=10)
            recovery_date = treatment.date + timedelta(days=18)
            nadir_wbc = max(2.0, 4.4 - (0.3 * index))
            recovery_wbc = min(5.5, nadir_wbc + 1.4)

            db.add(LabResult(
                patient_id=patient.id,
                date=nadir_date,
                wbc=round(nadir_wbc, 1),
                hemoglobin=round(12.5 - (0.25 * index), 1),
                platelets=max(135, 215 - (10 * index)),
                source="synthetic_qin_demo",
                source_note="Synthetic post-treatment CBC nadir for demo only.",
            ))
            db.add(LabResult(
                patient_id=patient.id,
                date=recovery_date,
                wbc=round(recovery_wbc, 1),
                hemoglobin=round(12.4 - (0.2 * index), 1),
                platelets=max(150, 225 - (8 * index)),
                source="synthetic_qin_demo",
                source_note="Synthetic CBC recovery value for demo only.",
            ))
            created += 2

    db.commit()
    return {
        "patients_found": len(patients),
        "patients_skipped": skipped_patients,
        "labs_created": created,
        "source": "synthetic_qin_demo",
    }


def patient_treatments(db, patient_id):
    from backend.models import Treatment

    return (
        db.query(Treatment)
        .filter(Treatment.patient_id == patient_id)
        .order_by(Treatment.date)
        .all()
    )
