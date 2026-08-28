"""
Patient router — /me/*, /patients/*, /patient-report/*, /summary-reviews, /clinician/review-queue.

Includes the streaming chat endpoint at POST /me/chat/stream and
POST /patients/{patient_id}/chat/stream.
"""

from __future__ import annotations


from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.api.deps import (
    get_db,
    get_patient_access_context,
)
from backend.api.schemas.patient import (
    MyImagingReportCreate,
    MyLabCreate,
    MyMedicationCreate,
    MySymptomCreate,
    MyTreatmentCreate,
)
from backend.crud import (
    get_patient,
)
from backend.models import (
    ImagingReport,
    LabResult,
    MedicationLog,
    SymptomReport,
    Treatment,
)
from backend.services.app_logging import log_app_event
from backend.services.input_validation import (
    validate_cbc_values,
    validate_imaging_report_payload,
    validate_symptom_payload,
    validate_treatment_payload,
    validation_error_payload,
)
from backend.services.ctcae_mapping import map_symptom_to_ctcae_review_hint
from backend.services.lab_reference_context import build_cbc_reference_context

from backend.api.routers.patient import _invalidate_report_cache


# No `tags=` here on purpose. This router is mounted only by
# `backend/api/routers/patient.py`, whose own router already declares
# `tags=["patient"]`, and FastAPI concatenates the parent's tags with the
# child's. Declaring the tag in both places emitted `["patient", "patient"]`
# for all 22 operations in this module, which is what the committed OpenAPI
# schema (single tag) shows was intended.
router = APIRouter()


@router.post("/me/symptoms")
def add_my_symptom(
    payload: MySymptomCreate,
    context=Depends(get_patient_access_context),
    db: Session = Depends(get_db),
):
    """Patient-scoped symptom save (manual-entry form).

    Mirrors the clinician-side :func:`add_symptom_report` route but trusts
    the bearer-token's patient_id rather than reading it from the URL.  The
    new ``duration`` and ``urgent_flag`` fields are folded into the notes
    column so the existing schema does not need to change — the clinician
    review queue picks the urgent tag up via its existing text scan.
    """
    patient_id = context.patient_id
    if not patient_id:
        raise HTTPException(status_code=403, detail="Patient context required.")
    if not get_patient(db, patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")

    composed_notes_parts: list[str] = []
    if payload.urgent_flag:
        composed_notes_parts.append("[urgent flag set by patient]")
    if payload.duration:
        composed_notes_parts.append(f"Duration: {payload.duration.strip()}")
    if payload.notes:
        composed_notes_parts.append(payload.notes.strip())
    composed_notes = " | ".join(composed_notes_parts) or None
    ctcae_hint = map_symptom_to_ctcae_review_hint(
        symptom=payload.symptom,
        severity=payload.severity,
        urgent_flag=payload.urgent_flag,
        notes=composed_notes,
    )

    try:
        validation_warnings = validate_symptom_payload(
            payload.symptom, payload.severity, composed_notes
        )
    except ValueError as exc:
        log_app_event(
            db=db,
            event_type="validation_error",
            patient_id=patient_id,
            route="/me/symptoms",
            status="error",
            input_payload=payload.dict(),
            error_message=str(exc),
        )
        raise HTTPException(
            status_code=400,
            detail=validation_error_payload(exc, route="/me/symptoms"),
        ) from exc

    symptom = SymptomReport(
        patient_id=patient_id,
        date=payload.date,
        symptom=payload.symptom,
        severity=payload.severity,
        notes=composed_notes,
    )
    db.add(symptom)
    db.commit()
    _invalidate_report_cache(patient_id)
    log_app_event(
        db=db,
        event_type="patient_input",
        patient_id=patient_id,
        route="/me/symptoms",
        status="ok",
        input_payload=payload.dict(),
        output_payload={
            "symptom_id": symptom.id,
            "warning_count": len(validation_warnings),
            "urgent_flag": payload.urgent_flag,
            "ctcae_hint": ctcae_hint,
        },
    )
    return {
        "message": "Symptom logged to your patient record.",
        "symptom_id": symptom.id,
        "validation_warnings": validation_warnings,
        "urgent_flag": payload.urgent_flag,
        "ctcae_review_hint": ctcae_hint,
        "safety_note": (
            "This record is for monitoring only and does not replace clinician judgement. "
            "Urgent symptoms should be discussed with your care team."
        ),
    }


@router.post("/me/labs")
def add_my_lab(
    payload: MyLabCreate,
    context=Depends(get_patient_access_context),
    db: Session = Depends(get_db),
):
    """Patient-scoped CBC save.  Mirrors :func:`add_lab_result` but trusts
    the bearer-token's patient_id rather than the URL.  ``anc`` and
    ``lab_source``/``notes`` are folded into ``source_note`` to avoid a
    schema migration on the LabResult table."""
    patient_id = context.patient_id
    if not patient_id:
        raise HTTPException(status_code=403, detail="Patient context required.")
    if not get_patient(db, patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")

    try:
        validation_warnings = validate_cbc_values(
            payload.wbc, payload.hemoglobin, payload.platelets
        )
    except ValueError as exc:
        log_app_event(
            db=db,
            event_type="validation_error",
            patient_id=patient_id,
            route="/me/labs",
            status="error",
            input_payload=payload.dict(),
            error_message=str(exc),
        )
        raise HTTPException(
            status_code=400,
            detail=validation_error_payload(exc, route="/me/labs"),
        ) from exc

    note_parts: list[str] = []
    if payload.anc is not None:
        note_parts.append(f"ANC={payload.anc:g} K/uL")
    if payload.lab_source:
        note_parts.append(f"Source: {payload.lab_source.strip()}")
    if payload.notes:
        note_parts.append(payload.notes.strip())
    source_note = " | ".join(note_parts) or None
    reference_context = build_cbc_reference_context(
        wbc=payload.wbc,
        hemoglobin=payload.hemoglobin,
        platelets=payload.platelets,
    )

    lab = LabResult(
        patient_id=patient_id,
        date=payload.date,
        wbc=payload.wbc,
        hemoglobin=payload.hemoglobin,
        platelets=payload.platelets,
        source="manual",
        source_note=source_note,
    )
    db.add(lab)
    db.commit()
    _invalidate_report_cache(patient_id)
    log_app_event(
        db=db,
        event_type="patient_input",
        patient_id=patient_id,
        route="/me/labs",
        status="ok",
        input_payload=payload.dict(),
        output_payload={
            "lab_id": lab.id,
            "warning_count": len(validation_warnings),
            "reference_context": reference_context,
        },
    )
    return {
        "message": "Lab values logged to your patient record.",
        "lab_id": lab.id,
        "validation_warnings": validation_warnings,
        "reference_context": reference_context,
        "safety_note": (
            "Lab entries here are for tracking. Reference ranges shown in the portal "
            "are general guides — your clinical team uses lab-specific ranges."
        ),
    }


@router.post("/me/imaging-reports")
def add_my_imaging_report(
    payload: MyImagingReportCreate,
    context=Depends(get_patient_access_context),
    db: Session = Depends(get_db),
):
    """Patient-scoped imaging save.  Either ``findings`` or ``impression``
    must be present; otherwise the entry is rejected so we never store an
    empty imaging row."""
    patient_id = context.patient_id
    if not patient_id:
        raise HTTPException(status_code=403, detail="Patient context required.")
    if not get_patient(db, patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")

    findings = (payload.findings or "").strip()
    impression = (payload.impression or "").strip()
    if not findings and not impression:
        raise HTTPException(
            status_code=400,
            detail="Please paste either the report findings or the impression text.",
        )
    body_site = (payload.body_site or "Breast").strip() or "Breast"
    report_type = (payload.report_type or "Patient-entered report").strip()

    try:
        validation_warnings = validate_imaging_report_payload(
            payload.modality,
            report_type,
            findings or impression,
            impression or findings,
            body_site,
        )
    except ValueError as exc:
        log_app_event(
            db=db,
            event_type="validation_error",
            patient_id=patient_id,
            route="/me/imaging-reports",
            status="error",
            input_payload=payload.dict(),
            error_message=str(exc),
        )
        raise HTTPException(
            status_code=400,
            detail=validation_error_payload(exc, route="/me/imaging-reports"),
        ) from exc

    composed_impression = impression
    if payload.notes:
        composed_impression = (
            f"{composed_impression}\n[Patient note] {payload.notes.strip()}".strip()
        )

    report = ImagingReport(
        patient_id=patient_id,
        date=payload.date,
        modality=payload.modality,
        report_type=report_type,
        body_site=body_site,
        findings=findings,
        impression=composed_impression,
    )
    db.add(report)
    db.commit()
    _invalidate_report_cache(patient_id)
    log_app_event(
        db=db,
        event_type="patient_input",
        patient_id=patient_id,
        route="/me/imaging-reports",
        status="ok",
        # Redact the free-text fields from the audit log — keep only metadata.
        input_payload={
            **payload.dict(),
            "findings": "[redacted report text]",
            "impression": "[redacted report text]",
        },
        output_payload={"imaging_report_id": report.id, "warning_count": len(validation_warnings)},
    )
    return {
        "message": "Imaging report logged to your patient record.",
        "imaging_report_id": report.id,
        "modality": payload.modality,
        "validation_warnings": validation_warnings,
        "safety_note": (
            "Imaging text is recorded as-is; this system does not interpret images. "
            "Your care team makes any clinical decisions."
        ),
    }


@router.post("/me/medications")
def add_my_medication(
    payload: MyMedicationCreate,
    context=Depends(get_patient_access_context),
    db: Session = Depends(get_db),
):
    """Patient-scoped medication save.  Stored in the MedicationLog table
    (the same table the chat agent's save_medication path writes to)."""
    patient_id = context.patient_id
    if not patient_id:
        raise HTTPException(status_code=403, detail="Patient context required.")
    if not get_patient(db, patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")

    medication = payload.medication.strip()
    if not medication:
        raise HTTPException(status_code=400, detail="Medication name is required.")
    if len(medication) > 120:
        raise HTTPException(
            status_code=400, detail="Medication name must be 120 characters or less."
        )

    note_parts: list[str] = []
    if payload.side_effects:
        note_parts.append(f"Side effects: {payload.side_effects.strip()}")
    if payload.notes:
        note_parts.append(payload.notes.strip())
    notes = " | ".join(note_parts) or None
    from backend.services.medication_interactions import check_medication_interactions

    current_medications = [
        row.medication
        for row in db.query(MedicationLog)
        .filter(MedicationLog.patient_id == patient_id)
        .order_by(MedicationLog.date.desc(), MedicationLog.id.desc())
        .limit(25)
        .all()
    ]
    interaction_check = check_medication_interactions(
        medication,
        current_medications=current_medications,
        notes=notes,
    )

    med = MedicationLog(
        patient_id=patient_id,
        date=payload.date,
        medication=medication,
        dose=(payload.dose or "").strip() or None,
        frequency=(payload.frequency or "").strip() or None,
        notes=notes,
    )
    db.add(med)
    db.commit()
    _invalidate_report_cache(patient_id)
    log_app_event(
        db=db,
        event_type="patient_input",
        patient_id=patient_id,
        route="/me/medications",
        status="ok",
        input_payload=payload.dict(),
        output_payload={"medication_id": med.id},
    )
    return {
        "message": "Medication logged to your patient record.",
        "medication_id": med.id,
        "safety_note": (
            "Use this to track what you are taking. Dose changes must be agreed with your care team."
        ),
        "interaction_check": interaction_check,
    }


@router.post("/me/treatments")
def add_my_treatment(
    payload: MyTreatmentCreate,
    context=Depends(get_patient_access_context),
    db: Session = Depends(get_db),
):
    """Patient-scoped treatment-cycle note.  Cycle defaults to 0 when the
    patient doesn't remember the number; the existing Treatment row schema
    requires an integer."""
    patient_id = context.patient_id
    if not patient_id:
        raise HTTPException(status_code=403, detail="Patient context required.")
    if not get_patient(db, patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")

    drug = payload.drug.strip()
    if not drug:
        raise HTTPException(status_code=400, detail="Treatment/drug name is required.")
    cycle = payload.cycle if payload.cycle is not None else 0
    try:
        validation_warnings = validate_treatment_payload(cycle, drug)
    except ValueError as exc:
        log_app_event(
            db=db,
            event_type="validation_error",
            patient_id=patient_id,
            route="/me/treatments",
            status="error",
            input_payload=payload.dict(),
            error_message=str(exc),
        )
        raise HTTPException(
            status_code=400,
            detail=validation_error_payload(exc, route="/me/treatments"),
        ) from exc

    treatment = Treatment(
        patient_id=patient_id,
        date=payload.date,
        cycle=cycle,
        drug=drug,
    )
    db.add(treatment)
    db.commit()
    _invalidate_report_cache(patient_id)
    log_app_event(
        db=db,
        event_type="patient_input",
        patient_id=patient_id,
        route="/me/treatments",
        status="ok",
        input_payload=payload.dict(),
        output_payload={
            "treatment_id": treatment.id,
            "warning_count": len(validation_warnings),
        },
    )
    return {
        "message": "Treatment note logged to your patient record.",
        "treatment_id": treatment.id,
        "validation_warnings": validation_warnings,
        "safety_note": (
            "This is a tracking note. Treatment decisions stay with your oncology team."
        ),
    }
