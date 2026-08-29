"""Patient timeline assembly.

The timeline merges lab CBC entries, treatment cycles, imaging report
summaries, symptom reports, and AI risk flags into a chronologically
ordered list of events for the patient view.

Risk events (produced by the deterministic risk engine) are tagged as
``ai_generated`` and carry the uncertainty block from the risk evidence
so the frontend ``AIGeneratedLabel`` can render confidence, uncertainty
reason, missing-data indicators, and the "Clinician review required"
amber badge. See [SAFETY_CARD.md](../../SAFETY_CARD.md) for why every
AI/model output must surface these fields together.
"""

from pathlib import Path
from typing import Any


def build_clinical_timeline(labs, treatments, imaging_reports, symptoms, risks, media_records=None):
    media_records = media_records or []

    events = []
    seen = set()

    def add_event(event):
        key = (
            str(event.get("date", ""))[:10],
            event.get("type"),
            event.get("title"),
            event.get("summary"),
        )
        if key in seen:
            return
        seen.add(key)
        events.append(event)

    if labs is not None and not labs.empty:
        for _, row in labs.iterrows():
            add_event({
                "date": str(row["date"]),
                "type": "lab",
                "title": "CBC result",
                "summary": (
                    f"WBC {row['wbc']}, hemoglobin {row['hemoglobin']}, "
                    f"platelets {row['platelets']}"
                ),
                "detail": {
                    "kind": "lab",
                    "title": "CBC result",
                    "fields": {
                        "WBC": row.get("wbc"),
                        "Hemoglobin": row.get("hemoglobin"),
                        "Platelets": row.get("platelets"),
                    },
                    "notes": "CBC values are shown for monitoring context only and require care-team interpretation.",
                },
                "ai_generated": False,
                "evidence_source": "lab_record",
            })

    if treatments is not None and not treatments.empty:
        for _, row in treatments.iterrows():
            add_event({
                "date": str(row["date"]),
                "type": "treatment",
                "title": f"Treatment cycle {row['cycle']}",
                "summary": str(row["drug"]),
                "detail": {
                    "kind": "treatment",
                    "title": f"Treatment cycle {row.get('cycle')}",
                    "fields": {
                        "Cycle": row.get("cycle"),
                        "Drug/regimen": row.get("drug"),
                    },
                    "notes": "Treatment entries document timeline context; the system does not recommend dose or regimen changes.",
                },
                "ai_generated": False,
                "evidence_source": "treatment_record",
            })

    if imaging_reports is not None and not imaging_reports.empty:
        for _, row in imaging_reports.iterrows():
            modality = row.get("modality", "Imaging")
            add_event({
                "date": str(row["date"]),
                "type": "imaging",
                "title": f"{modality} - {row['report_type']}",
                "summary": row["impression"],
                "detail": {
                    "kind": "imaging",
                    "title": f"{modality} - {row.get('report_type')}",
                    "fields": {
                        "Modality": modality,
                        "Report type": row.get("report_type"),
                        "Body site": row.get("body_site"),
                    },
                    "findings": row.get("findings"),
                    "impression": row.get("impression"),
                    "media": _matching_media(media_records, row),
                    "notes": "Imaging report text and preview files are for clinician review. NLCare does not diagnose response, recurrence, or metastasis from images or wording.",
                },
                "ai_generated": False,
                "evidence_source": "imaging_report",
            })

    if symptoms is not None and not symptoms.empty:
        for _, row in symptoms.iterrows():
            note = f" - {row['notes']}" if row.get("notes") else ""
            add_event({
                "date": str(row["date"]),
                "type": "symptom",
                "title": f"Symptom: {row['symptom']}",
                "summary": f"Severity {row['severity']}/10{note}",
                "detail": {
                    "kind": "symptom",
                    "title": f"Symptom: {row.get('symptom')}",
                    "fields": {
                        "Symptom": row.get("symptom"),
                        "Severity": f"{row.get('severity')}/10",
                    },
                    "notes": row.get("notes") or "No notes recorded.",
                },
                "ai_generated": False,
                "evidence_source": "patient_report",
            })

    for risk in risks:
        evidence = risk.get("evidence") or {}
        risk_date = evidence.get("date")
        if risk_date:
            add_event({
                "date": str(risk_date),
                "type": "ai_risk_flag",
                "title": f"Risk flag: {risk.get('type')}",
                "summary": risk.get("message"),
                "severity": risk.get("severity"),
                "detail": {
                    "kind": "risk_flag",
                    "title": f"Risk flag: {risk.get('type')}",
                    "fields": {
                        "Severity": risk.get("severity"),
                        "Category": risk.get("category"),
                        "Source": "deterministic risk engine",
                    },
                    "message": risk.get("message"),
                    "evidence": evidence,
                    "notes": "Risk flags are deterministic monitoring signals for clinician review, not diagnoses.",
                },
                "ai_generated": True,
                "evidence_source": "risk_engine",
                "model_version": evidence.get("threshold_config_version"),
                "uncertainty": risk.get("uncertainty"),
            })

    return sorted(events, key=lambda event: event["date"])


def _matching_media(media_records, row):
    event_date = str(row.get("date", ""))[:10]
    modality = str(row.get("modality") or "").lower()
    matches_by_path: dict[str, dict[str, Any]] = {}
    for record in media_records:
        record_date = str(record.get("scan_date") or record.get("created_at") or "")[:10]
        record_modality = str(record.get("modality") or record.get("upload_type") or "").lower()
        if event_date and record_date and event_date != record_date:
            continue
        if modality and record_modality and not _modality_overlap(modality, record_modality):
            continue
        path = record.get("local_path") or record.get("folder")
        upload_id = record.get("id") if record.get("original_filename") else None
        candidate = {
            "label": record.get("series_description") or record.get("original_filename") or record.get("modality") or "Uploaded file",
            "modality": record.get("modality") or record.get("upload_type"),
            "upload_id": upload_id,
            "artifact_url": f"/me/uploads/{upload_id}/content" if upload_id is not None else None,
            "content_type": record.get("content_type"),
            "previewable": _is_previewable(path, record.get("content_type")),
            "notes": record.get("notes"),
        }
        key = str(path or candidate["label"])
        existing = matches_by_path.get(key)
        if existing is None or (candidate["artifact_url"] and not existing.get("artifact_url")):
            matches_by_path[key] = candidate
    return list(matches_by_path.values())[:6]


def _modality_overlap(left, right):
    groups = [
        {"mri", "breast mri", "mr"},
        {"ct", "ct scan", "cat scan", "pet/ct", "pet-ct", "fdg pet/ct", "ct abdomen/pelvis", "ct chest"},
        {"ultrasound", "us", "sonogram", "breast ultrasound", "abdominal ultrasound"},
        {"mammogram", "mammography"},
    ]
    left_tokens = {left, *left.replace("/", " ").replace("-", " ").split()}
    right_tokens = {right, *right.replace("/", " ").replace("-", " ").split()}
    for group in groups:
        if left_tokens & group and right_tokens & group:
            return True
    return bool(left_tokens & right_tokens)


def _is_previewable(path, content_type):
    if content_type and str(content_type).lower().startswith("image/"):
        return True
    suffix = Path(str(path or "")).suffix.lower()
    return suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
