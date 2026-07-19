import base64
import json
from datetime import datetime, timezone
from pathlib import Path
import re

from backend.config import UPLOAD_DIR, ensure_runtime_dirs
from backend.models import MRIFileRegistry, PatientUpload
from backend.services.breast_cancer_journey import infer_journey_phase
from backend.services.medical_report_parser import parse_report


MAX_UPLOAD_BYTES = 12 * 1024 * 1024
IMAGING_UPLOAD_TYPES = {
    "mri",
    "breast_mri",
    "imaging",
    "ct",
    "ct_scan",
    "cat_scan",
    "pet_ct",
    "pet-ct",
    "ultrasound",
    "us",
    "sonogram",
    "breast_ultrasound",
    "abdominal_ultrasound",
}


def save_patient_upload(db, patient_id, upload_type, file_name, content_type, content_base64, notes=None, scan_date=None):
    ensure_runtime_dirs()
    decoded = _decode_base64_payload(content_base64)
    if len(decoded) > MAX_UPLOAD_BYTES:
        raise ValueError("Upload is too large for the demo endpoint")

    safe_type = _safe_segment(upload_type or "document")
    safe_name = _safe_filename(file_name)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    patient_dir = UPLOAD_DIR / _safe_segment(patient_id) / safe_type
    patient_dir.mkdir(parents=True, exist_ok=True)
    output_path = patient_dir / f"{timestamp}_{safe_name}"
    output_path.write_bytes(decoded)

    upload = PatientUpload(
        patient_id=patient_id,
        upload_type=safe_type,
        original_filename=file_name,
        content_type=content_type,
        local_path=str(output_path),
        notes=notes,
    )
    db.add(upload)

    if safe_type in IMAGING_UPLOAD_TYPES:
        db.add(MRIFileRegistry(
            patient_id=patient_id,
            scan_date=scan_date,
            modality=_infer_upload_modality(safe_type, file_name, notes),
            series_description=file_name,
            local_path=str(output_path),
            notes=notes or "Uploaded from patient portal.",
        ))

    db.commit()
    db.refresh(upload)
    return upload_to_dict(upload)


def get_patient_uploads(db, patient_id, limit=50):
    rows = (
        db.query(PatientUpload)
        .filter(PatientUpload.patient_id == patient_id)
        .order_by(PatientUpload.created_at.desc(), PatientUpload.id.desc())
        .limit(limit)
        .all()
    )
    return [upload_to_dict(row) for row in rows]


def upload_to_dict(row):
    parsed_report = _parse_upload_metadata(row)
    return {
        "id": row.id,
        "patient_id": row.patient_id,
        "upload_type": row.upload_type,
        "original_filename": row.original_filename,
        "content_type": row.content_type,
        "content_url": f"/me/uploads/{row.id}/content",
        "notes": row.notes,
        "parsed_report": parsed_report,
        "journey_phase": infer_journey_phase({
            "upload_type": row.upload_type,
            "modality": parsed_report.get("report_type") if isinstance(parsed_report, dict) else row.upload_type,
            "notes": row.notes,
        }),
        "created_at": str(row.created_at),
    }


def _decode_base64_payload(payload):
    if "," in payload:
        payload = payload.split(",", 1)[1]
    return base64.b64decode(payload, validate=False)


def _safe_segment(value):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return cleaned or "item"


def _safe_filename(value):
    name = Path(value or "upload.bin").name
    cleaned = re.sub(r"[^A-Za-z0-9_. -]+", "_", name).strip(" .")
    return cleaned or "upload.bin"


def _infer_upload_modality(upload_type, file_name, notes):
    text = f"{upload_type} {file_name or ''} {notes or ''}".lower()
    if "pet_ct" in text or "pet-ct" in text or "pet/ct" in text or "pet ct" in text:
        return "FDG PET/CT"
    if "cat_scan" in text or "cat scan" in text or re.search(r"\bct\b", text):
        if any(term in text for term in ["abdomen", "abdominal", "pelvis", "liver", "ascites", "peritoneal"]):
            return "CT abdomen/pelvis"
        if any(term in text for term in ["chest", "lung", "pleural", "mediastinal"]):
            return "CT chest"
        return "CT scan"
    if "ultrasound" in text or "sonogram" in text or text.startswith("us "):
        if any(term in text for term in ["abdomen", "abdominal", "liver", "ascites", "peritoneal"]):
            return "Abdominal ultrasound"
        return "Breast ultrasound"
    if "mammogram" in text or "mammography" in text:
        return "Mammogram"
    return "Breast MRI"


def _parse_upload_metadata(row):
    text = _extract_text_for_parser(row.local_path, row.content_type, row.original_filename)
    if row.notes:
        text = f"{text}\n\nUpload notes:\n{row.notes}" if text else row.notes
    if not text:
        return {
            "report_type": "non_text_upload",
            "parser_version": "medical_report_parser_v1_2026_05",
            "fields": {},
            "confidence": "not_parsed",
            "requires_review": True,
            "claim_boundary": "Upload stored only. Non-text/image/PDF files require manual clinician review or explicit OCR before structured extraction.",
        }
    try:
        return parse_report(text, filename=row.original_filename)
    except Exception as exc:
        return {
            "report_type": "parse_error",
            "parser_version": "medical_report_parser_v1_2026_05",
            "fields": {},
            "confidence": "not_parsed",
            "requires_review": True,
            "error": str(exc),
            "claim_boundary": "Parser failure does not imply absence of clinical information. The uploaded record should be reviewed manually.",
        }


def _extract_text_for_parser(local_path, content_type, file_name):
    path = Path(local_path or "")
    if not path.exists() or not path.is_file():
        return ""
    content_type = (content_type or "").lower()
    suffix = (Path(file_name or path.name).suffix or "").lower()
    if content_type.startswith("text/") or suffix in {".txt", ".md", ".csv", ".tsv", ".json"}:
        raw = path.read_bytes()[:200_000]
        for encoding in ("utf-8", "utf-16", "latin-1"):
            try:
                text = raw.decode(encoding)
                if suffix == ".json":
                    return json.dumps(json.loads(text), ensure_ascii=True)
                return text
            except Exception:
                continue
    return ""
