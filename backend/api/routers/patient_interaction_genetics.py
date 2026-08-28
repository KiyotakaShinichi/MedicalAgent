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
    get_clinician_or_admin_context,
)
from backend.api.schemas.patient import (
    BiomarkerRecordCreate,
    FamilyHistoryCreate,
    GeneticReviewCreate,
    GeneticTestRecordCreate,
    TumorMarkerRecordCreate,
)
from backend.crud import (
    get_patient,
)

from backend.api.routers.patient import _invalidate_report_cache


# No `tags=` here on purpose. This router is mounted only by
# `backend/api/routers/patient.py`, whose own router already declares
# `tags=["patient"]`, and FastAPI concatenates the parent's tags with the
# child's. Declaring the tag in both places emitted `["patient", "patient"]`
# for all 22 operations in this module, which is what the committed OpenAPI
# schema (single tag) shows was intended.
router = APIRouter()


@router.post("/me/family-history")
def add_my_family_history(
    payload: FamilyHistoryCreate,
    context=Depends(get_patient_access_context),
    db: Session = Depends(get_db),
):
    from backend.services.genetic_counseling import (
        GENETIC_BOUNDARY_NOTE,
        create_family_history_record,
    )

    record = create_family_history_record(
        db, context.patient_id, payload.dict(), actor_role=context.role
    )
    _invalidate_report_cache(context.patient_id)
    return {
        "message": "Family history saved for review.",
        "record": record,
        "safety_note": GENETIC_BOUNDARY_NOTE,
    }


@router.post("/me/genetic-test-records")
def add_my_genetic_test_record(
    payload: GeneticTestRecordCreate,
    context=Depends(get_patient_access_context),
    db: Session = Depends(get_db),
):
    from backend.services.genetic_counseling import (
        GENETIC_BOUNDARY_NOTE,
        create_genetic_test_record,
    )

    record = create_genetic_test_record(
        db, context.patient_id, payload.dict(), actor_role=context.role
    )
    _invalidate_report_cache(context.patient_id)
    return {
        "message": "Genetic test record saved for genetics/oncology review.",
        "record": record,
        "safety_note": GENETIC_BOUNDARY_NOTE,
    }


@router.post("/me/biomarker-records")
def add_my_biomarker_record(
    payload: BiomarkerRecordCreate,
    context=Depends(get_patient_access_context),
    db: Session = Depends(get_db),
):
    from backend.services.genetic_counseling import GENETIC_BOUNDARY_NOTE, create_biomarker_record

    record = create_biomarker_record(
        db, context.patient_id, payload.dict(), actor_role=context.role
    )
    _invalidate_report_cache(context.patient_id)
    return {
        "message": "Biomarker/pathology record saved for review.",
        "record": record,
        "safety_note": GENETIC_BOUNDARY_NOTE,
    }


@router.post("/me/tumor-marker-records")
def add_my_tumor_marker_record(
    payload: TumorMarkerRecordCreate,
    context=Depends(get_patient_access_context),
    db: Session = Depends(get_db),
):
    from backend.services.genetic_counseling import (
        GENETIC_BOUNDARY_NOTE,
        create_tumor_marker_record,
    )

    record = create_tumor_marker_record(
        db, context.patient_id, payload.dict(), actor_role=context.role
    )
    _invalidate_report_cache(context.patient_id)
    return {
        "message": "Tumor marker record saved for review.",
        "record": record,
        "safety_note": GENETIC_BOUNDARY_NOTE,
    }


@router.post("/patients/{patient_id}/genetic-counseling-review")
def save_genetic_counseling_review(
    patient_id: str,
    payload: GeneticReviewCreate,
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    from backend.services.genetic_counseling import create_genetic_review_note

    if not get_patient(db, patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")
    record = create_genetic_review_note(
        db, patient_id, context.role, payload.decision, payload.notes
    )
    _invalidate_report_cache(patient_id)
    return {"message": "Genetic counseling readiness review saved.", "review": record}
