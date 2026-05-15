from __future__ import annotations

import json
from datetime import date
from typing import Any

from sqlalchemy.orm import Session

from backend.models import (
    BiomarkerRecord,
    FamilyCancerHistoryRecord,
    GeneticCounselingReviewNote,
    GeneticTestRecord,
    TumorMarkerRecord,
)
from backend.services.app_logging import log_app_event


GENETIC_BOUNDARY_NOTE = (
    "Genetic Counseling Readiness organizes information for oncology/genetics review. "
    "It does not diagnose inherited risk, interpret variants as medical advice, or recommend treatment changes."
)

GENETIC_UNSAFE_PHRASES = [
    "you have brca",
    "you will get cancer",
    "your relatives will get cancer",
    "you should change chemo",
    "you should stop treatment",
    "you should start treatment",
    "this mutation means your treatment must",
    "no need for a genetic counselor",
    "vus means positive",
    "tumor marker proves cancer",
]


def build_genetic_counseling_readiness(db: Session, patient_id: str) -> dict[str, Any]:
    family = _family_rows(db, patient_id)
    genetic_tests = _genetic_test_rows(db, patient_id)
    biomarkers = _biomarker_rows(db, patient_id)
    tumor_markers = _tumor_marker_rows(db, patient_id)
    reviews = _review_rows(db, patient_id)

    flags = _readiness_flags(family, genetic_tests, biomarkers, tumor_markers)
    missing = []
    if not family:
        missing.append("No family cancer history recorded.")
    if not genetic_tests:
        missing.append("No genetic test record available.")
    if not biomarkers:
        missing.append("No biomarker/pathology record available.")
    if not tumor_markers:
        missing.append("No tumor marker trend records available.")

    return {
        "schema_version": "genetic_counseling_readiness_v1",
        "patient_id": patient_id,
        "boundary_note": GENETIC_BOUNDARY_NOTE,
        "family_history": family,
        "genetic_test_records": genetic_tests,
        "biomarker_records": biomarkers,
        "tumor_marker_records": tumor_markers,
        "review_notes": reviews,
        "flags": flags,
        "missing_data": missing,
        "readiness_status": _readiness_status(flags, missing),
        "questions_to_ask": [
            "Would genetic counseling be useful based on my personal and family history?",
            "Is any existing genetic result germline or tumor/somatic testing?",
            "Does any VUS need follow-up or reclassification review?",
            "How should biomarker or tumor marker results be interpreted in my full clinical context?",
        ],
    }


def create_family_history_record(db: Session, patient_id: str, payload: dict[str, Any], *, actor_role: str = "patient"):
    record = FamilyCancerHistoryRecord(patient_id=patient_id, **payload)
    db.add(record)
    db.commit()
    db.refresh(record)
    _audit(db, "family_history_created", patient_id, actor_role, payload, {"record_id": record.id})
    return _family_to_dict(record)


def create_genetic_test_record(db: Session, patient_id: str, payload: dict[str, Any], *, actor_role: str = "patient"):
    record = GeneticTestRecord(patient_id=patient_id, **payload)
    db.add(record)
    db.commit()
    db.refresh(record)
    _audit(db, "genetic_test_record_created", patient_id, actor_role, payload, {"record_id": record.id})
    return _genetic_test_to_dict(record)


def create_biomarker_record(db: Session, patient_id: str, payload: dict[str, Any], *, actor_role: str = "patient"):
    record = BiomarkerRecord(patient_id=patient_id, **payload)
    db.add(record)
    db.commit()
    db.refresh(record)
    _audit(db, "biomarker_record_created", patient_id, actor_role, payload, {"record_id": record.id})
    return _biomarker_to_dict(record)


def create_tumor_marker_record(db: Session, patient_id: str, payload: dict[str, Any], *, actor_role: str = "patient"):
    record = TumorMarkerRecord(patient_id=patient_id, **payload)
    db.add(record)
    db.commit()
    db.refresh(record)
    _audit(db, "tumor_marker_record_created", patient_id, actor_role, payload, {"record_id": record.id})
    return _tumor_marker_to_dict(record)


def create_genetic_review_note(db: Session, patient_id: str, reviewer_role: str, decision: str, notes: str | None):
    snapshot = build_genetic_counseling_readiness(db, patient_id)
    record = GeneticCounselingReviewNote(
        patient_id=patient_id,
        reviewer_role=reviewer_role,
        decision=decision,
        notes=notes,
        readiness_snapshot_json=json.dumps(snapshot),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    _audit(db, "genetic_counseling_review_saved", patient_id, reviewer_role, {"decision": decision}, {"record_id": record.id})
    return _review_to_dict(record)


def genetics_intent_and_safe_reply(query: str) -> dict[str, Any]:
    lower = query.lower()
    intent = "genetic_counseling_question"
    if "vus" in lower or "variant of uncertain" in lower:
        intent = "vus_explanation"
        reply = (
            "A VUS means variant of uncertain significance. It is not the same as a confirmed harmful mutation "
            "and should not be used by itself to make treatment or prevention decisions. This should be reviewed "
            "with a genetics-trained clinician or genetic counselor."
        )
    elif "germline" in lower or "somatic" in lower:
        intent = "germline_vs_somatic"
        reply = (
            "Germline testing looks for inherited DNA changes, often from blood or saliva. Somatic or tumor testing "
            "looks for changes in cancer cells or tumor tissue. Both need clinical context and should be reviewed "
            "by the oncology or genetics team."
        )
    elif any(marker in lower for marker in ["ca 15-3", "ca 27.29", "cea", "tumor marker"]):
        intent = "tumor_marker_explanation"
        reply = (
            "Tumor markers can be context-dependent monitoring clues, especially as trends, but they cannot diagnose "
            "recurrence or prove cancer by themselves. A high value should be reviewed by your oncology team with "
            "symptoms, imaging, labs, and treatment history."
        )
    elif any(term in lower for term in ["change chemo", "stop chemo", "start chemo", "what treatment", "should i take", "change surgery"]):
        intent = "treatment_decision_boundary"
        reply = (
            "I cannot recommend treatment changes based on genetics, biomarkers, or tumor markers. This record should "
            "be reviewed by your oncology team and, when relevant, a genetic counselor."
        )
    elif any(term in lower for term in ["sister", "brother", "mother", "father", "relative", "aunt", "uncle"]) and "upload" in lower:
        intent = "genetic_counseling_question"
        reply = (
            "Family genetic records can contain private information about another person. Please do not upload a "
            "relative's identifiable genetic test report unless you have permission and your care team says it is "
            "appropriate. You can record non-identifying family-history facts for clinician review."
        )
    elif any(term in lower for term in ["will i get", "do i have brca", "will my family", "will my relatives"]):
        intent = "risk_prediction_boundary"
        reply = (
            "I cannot predict whether you or relatives will develop cancer or determine BRCA status. Family history "
            "and testing questions are best reviewed with a genetics-trained clinician or genetic counselor."
        )
    elif any(term in lower for term in ["her2", "er", "pr", "ki-67", "ki67", "biomarker"]):
        intent = "biomarker_explanation"
        reply = (
            "ER, PR, HER2, and sometimes Ki-67 are tumor biomarkers reported from cancer cells in biopsy or surgery "
            "pathology. They help clinicians understand tumor biology and plan care, but this portal only organizes "
            "the record for review."
        )
    elif any(term in lower for term in ["brca", "palb2", "tp53", "pten", "chek2", "atm", "hereditary"]):
        intent = "hereditary_risk_question"
        reply = (
            "Some genes such as BRCA1, BRCA2, PALB2, TP53, PTEN, CHEK2, and ATM can be discussed in hereditary cancer "
            "risk assessment. This portal can organize questions and records, but a genetic counselor or qualified "
            "clinician should interpret testing and family-history implications."
        )
    else:
        reply = (
            "Genetic counseling can help people understand whether genetic testing may be useful and what results may "
            "mean for them and their family. OncoTrack can organize family history and test records for clinician review."
        )

    return {
        "intent": intent,
        "reply": f"{reply} {GENETIC_BOUNDARY_NOTE}",
        "citations_required": intent not in {"treatment_decision_boundary", "risk_prediction_boundary"},
        "review_required": True,
    }


def check_genetics_output_safety(reply: str) -> dict[str, Any]:
    lower = reply.lower()
    issues = [phrase for phrase in GENETIC_UNSAFE_PHRASES if phrase in lower]
    return {
        "passed": not issues,
        "issues": issues,
        "safe_alternatives": [
            "This may be worth discussing with a genetics-trained clinician or genetic counselor.",
            "This record should be reviewed by your oncology team.",
            "A VUS is uncertain and should not be treated like a confirmed harmful variant.",
            "Tumor markers are context-dependent and cannot diagnose cancer by themselves.",
        ],
    }


def _readiness_flags(family, genetic_tests, biomarkers, tumor_markers):
    flags = []
    if any(row.get("cancer_type") in {"ovarian", "pancreatic", "male breast", "prostate"} for row in family):
        flags.append("family_history_pattern_for_genetics_review")
    if any(str(row.get("known_familial_mutation")).lower() == "yes" for row in family):
        flags.append("known_familial_mutation_reported")
    if any((row.get("classification") or "").lower() in {"pathogenic", "likely pathogenic", "vus"} for row in genetic_tests):
        flags.append("genetic_test_result_needs_review")
    if any((row.get("classification") or "").lower() == "vus" for row in genetic_tests):
        flags.append("vus_should_not_be_treated_as_positive")
    if biomarkers:
        flags.append("biomarker_pathology_record_available")
    if tumor_markers:
        flags.append("tumor_marker_trend_requires_context")
    return flags


def _readiness_status(flags, missing):
    if flags:
        return "needs_genetics_or_oncology_review"
    if len(missing) >= 3:
        return "insufficient_information"
    return "organized_for_review"


def _family_rows(db, patient_id):
    return [_family_to_dict(row) for row in db.query(FamilyCancerHistoryRecord).filter_by(patient_id=patient_id).order_by(FamilyCancerHistoryRecord.created_at.desc()).all()]


def _genetic_test_rows(db, patient_id):
    return [_genetic_test_to_dict(row) for row in db.query(GeneticTestRecord).filter_by(patient_id=patient_id).order_by(GeneticTestRecord.created_at.desc()).all()]


def _biomarker_rows(db, patient_id):
    return [_biomarker_to_dict(row) for row in db.query(BiomarkerRecord).filter_by(patient_id=patient_id).order_by(BiomarkerRecord.created_at.desc()).all()]


def _tumor_marker_rows(db, patient_id):
    return [_tumor_marker_to_dict(row) for row in db.query(TumorMarkerRecord).filter_by(patient_id=patient_id).order_by(TumorMarkerRecord.date_collected.desc(), TumorMarkerRecord.id.desc()).all()]


def _review_rows(db, patient_id):
    return [_review_to_dict(row) for row in db.query(GeneticCounselingReviewNote).filter_by(patient_id=patient_id).order_by(GeneticCounselingReviewNote.created_at.desc()).limit(10).all()]


def _family_to_dict(row):
    return _row_dict(row, ["id", "relationship", "family_side", "cancer_type", "age_at_diagnosis", "relative_status", "multiple_relatives_affected", "male_breast_cancer", "known_familial_mutation", "notes", "review_status", "source", "created_at"])


def _genetic_test_to_dict(row):
    return _row_dict(row, ["id", "test_type", "sample_type", "gene", "variant_text", "classification", "report_date", "lab_provider", "upload_reference", "reviewed_by_genetic_counselor", "clinician_review_status", "notes", "created_at"])


def _biomarker_to_dict(row):
    return _row_dict(row, ["id", "source", "er_status", "pr_status", "her2_status", "ki67_percent", "grade", "stage", "report_date", "report_text", "upload_reference", "clinician_review_needed", "review_status", "created_at"])


def _tumor_marker_to_dict(row):
    return _row_dict(row, ["id", "marker", "value", "unit", "reference_range", "date_collected", "trend_direction", "notes", "review_status", "created_at"])


def _review_to_dict(row):
    return _row_dict(row, ["id", "reviewer_role", "decision", "notes", "created_at"])


def _row_dict(row, fields):
    out = {}
    for field in fields:
        value = getattr(row, field, None)
        if isinstance(value, date):
            value = value.isoformat()
        out[field] = value
    return out


def _audit(db, event_type, patient_id, actor_role, input_payload, output_payload):
    log_app_event(
        db=db,
        event_type=event_type,
        actor_role=actor_role,
        patient_id=patient_id,
        route="genetic_counseling_readiness",
        status="ok",
        input_payload=input_payload,
        output_payload=output_payload,
    )
