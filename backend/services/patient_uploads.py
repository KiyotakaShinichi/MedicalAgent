import base64
from datetime import datetime, timezone
from pathlib import Path
import re

from backend.config import UPLOAD_DIR, ensure_runtime_dirs
from backend.models import MRIFileRegistry, PatientUpload


MAX_UPLOAD_BYTES = 12 * 1024 * 1024


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

    if safe_type in {"mri", "breast_mri", "imaging"}:
        db.add(MRIFileRegistry(
            patient_id=patient_id,
            scan_date=scan_date,
            modality="Breast MRI",
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
    return {
        "id": row.id,
        "patient_id": row.patient_id,
        "upload_type": row.upload_type,
        "original_filename": row.original_filename,
        "content_type": row.content_type,
        "local_path": row.local_path,
        "notes": row.notes,
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
