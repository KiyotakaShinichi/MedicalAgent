from backend.models import LabResult, Treatment
from backend.services.synthetic_cbc import generate_synthetic_cbc_for_qin_patients

QIN_EPISODE_GAP_DAYS = 7


def sync_qin_treatment_cycles(db):
    """Merge same-day QIN drug-agent rows into coherent treatment cycles and refresh demo CBC rows."""

    patient_ids = [
        patient_id for (patient_id,) in (
            db.query(Treatment.patient_id)
            .filter(Treatment.patient_id.like("QIN-BREAST-02-%"))
            .distinct()
            .all()
        )
    ]
    patients_updated = 0
    rows_deleted = 0
    rows_inserted = 0

    for patient_id in patient_ids:
        treatments = (
            db.query(Treatment)
            .filter(Treatment.patient_id == patient_id)
            .order_by(Treatment.date, Treatment.cycle)
            .all()
        )
        if not treatments:
            continue
        merged = []
        for cycle, (treatment_date, drugs) in enumerate(_group_treatment_episodes(treatments), start=1):
            merged.append(Treatment(
                patient_id=patient_id,
                date=treatment_date,
                cycle=cycle,
                drug=_merge_drug_names(drugs),
            ))

        needs_update = (
            len(merged) != len(treatments)
            or any(old.cycle != new.cycle or old.date != new.date or old.drug != new.drug for old, new in zip(treatments, merged))
        )
        if not needs_update:
            continue

        rows_deleted += len(treatments)
        patients_updated += 1
        for treatment in treatments:
            db.delete(treatment)
        db.flush()
        db.add_all(merged)
        rows_inserted += len(merged)

        db.query(LabResult).filter(
            LabResult.patient_id == patient_id,
            LabResult.source == "synthetic_qin_demo",
        ).delete(synchronize_session=False)

    db.commit()
    cbc_result = generate_synthetic_cbc_for_qin_patients(db)
    return {
        "patients_seen": len(patient_ids),
        "patients_updated": patients_updated,
        "treatment_rows_deleted": rows_deleted,
        "treatment_rows_inserted": rows_inserted,
        "cbc_refresh": cbc_result,
        "source": "qin_treatment_cycle_sync",
    }


def _merge_drug_names(drugs):
    cleaned = []
    for drug in drugs:
        value = str(drug or "").strip()
        if value and value not in cleaned:
            cleaned.append(value)
    return " + ".join(cleaned) if cleaned else "unspecified regimen"


def _group_treatment_episodes(treatments):
    episodes = []
    current_date = None
    current_drugs = []
    last_date = None
    for row in sorted(treatments, key=lambda item: (item.date, item.cycle)):
        if current_date is None:
            current_date = row.date
            last_date = row.date
            current_drugs = [row.drug]
            continue
        if (row.date - last_date).days <= QIN_EPISODE_GAP_DAYS:
            current_drugs.append(row.drug)
            last_date = row.date
            continue
        episodes.append((current_date, current_drugs))
        current_date = row.date
        last_date = row.date
        current_drugs = [row.drug]
    if current_date is not None:
        episodes.append((current_date, current_drugs))
    return episodes
