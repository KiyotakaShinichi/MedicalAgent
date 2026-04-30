from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pydicom
from sqlalchemy.orm import Session

from backend.models import MRISeriesIndex, Patient


def index_mri_series(db: Session, root_path: str, patient_id: str | None = None, max_files: int | None = None):
    root = Path(root_path)
    if not root.exists():
        raise FileNotFoundError(f"MRI root path not found: {root_path}")

    if patient_id is None:
        subject_dirs = sorted(
            path for path in root.iterdir()
            if path.is_dir() and path.name.startswith("QIN-BREAST-02-")
        )
        if subject_dirs:
            aggregate = {
                "root_path": str(root),
                "subjects_indexed": 0,
                "files_seen": 0,
                "dicom_files_scanned": 0,
                "series_found": 0,
                "series_created": 0,
                "series_updated": 0,
                "series_skipped_missing_patient": 0,
                "limited_scan": max_files is not None,
                "candidate_roles": {},
            }

            for subject_dir in subject_dirs:
                result = index_mri_series(
                    db=db,
                    root_path=str(root),
                    patient_id=subject_dir.name,
                    max_files=max_files,
                )
                aggregate["subjects_indexed"] += 1
                for key in (
                    "files_seen",
                    "dicom_files_scanned",
                    "series_found",
                    "series_created",
                    "series_updated",
                    "series_skipped_missing_patient",
                ):
                    aggregate[key] += result[key]
                for role, count in result["candidate_roles"].items():
                    aggregate["candidate_roles"][role] = aggregate["candidate_roles"].get(role, 0) + count

            return aggregate

    scan_root = root
    if patient_id and (root / patient_id).is_dir():
        scan_root = root / patient_id

    file_paths = [path for path in scan_root.rglob("*") if path.is_file()]
    limited_scan = max_files is not None
    if max_files:
        file_paths = file_paths[:max_files]

    series = {}
    grouped = defaultdict(int)
    scanned_dicom_files = 0

    for path in file_paths:
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception:
            continue

        subject = str(getattr(dataset, "PatientID", "")) or _subject_from_path(path)
        if patient_id and subject != patient_id:
            continue

        scanned_dicom_files += 1
        series_uid = str(getattr(dataset, "SeriesInstanceUID", path.parent.name))
        grouped[series_uid] += 1

        if series_uid not in series:
            description = str(getattr(dataset, "SeriesDescription", "")) or None
            series[series_uid] = {
                "patient_id": subject,
                "study_date": _parse_dicom_date(str(getattr(dataset, "StudyDate", ""))),
                "modality": str(getattr(dataset, "Modality", "")) or None,
                "series_description": description,
                "series_uid": series_uid,
                "folder": str(path.parent),
                "candidate_role": classify_mri_series_role(description),
            }

    created = 0
    updated = 0
    skipped_missing_patient = 0

    for series_uid, item in series.items():
        if not db.query(Patient).filter(Patient.id == item["patient_id"]).first():
            skipped_missing_patient += 1
            continue

        existing = db.query(MRISeriesIndex).filter(MRISeriesIndex.series_uid == series_uid).first()
        if existing:
            existing.patient_id = item["patient_id"]
            existing.study_date = item["study_date"]
            existing.modality = item["modality"]
            existing.series_description = item["series_description"]
            existing.folder = item["folder"]
            if not limited_scan or grouped[series_uid] >= existing.instance_count:
                existing.instance_count = grouped[series_uid]
            existing.candidate_role = item["candidate_role"]
            updated += 1
        else:
            db.add(MRISeriesIndex(
                patient_id=item["patient_id"],
                study_date=item["study_date"],
                modality=item["modality"],
                series_description=item["series_description"],
                series_uid=series_uid,
                folder=item["folder"],
                instance_count=grouped[series_uid],
                candidate_role=item["candidate_role"],
            ))
            created += 1

    db.commit()

    return {
        "root_path": str(scan_root),
        "files_seen": len(file_paths),
        "dicom_files_scanned": scanned_dicom_files,
        "series_found": len(series),
        "series_created": created,
        "series_updated": updated,
        "series_skipped_missing_patient": skipped_missing_patient,
        "limited_scan": limited_scan,
        "candidate_roles": _role_counts(series.values()),
    }


def classify_mri_series_role(series_description: str | None):
    text = (series_description or "").lower()

    if "dynamic" in text or "dce" in text or "dyn" in text:
        return "dce"
    if "dwi" in text or "diffusion" in text or "epi" in text or "b0" in text or "b200" in text or "b800" in text:
        return "dwi"
    if "thrive" in text or "t1" in text:
        return "t1w"
    if "qmt" in text or "magnetization transfer" in text:
        return "qmt"
    if "bloch" in text or "b1" in text:
        return "b1"
    if "b0" in text:
        return "b0"
    return "unknown"


def _parse_dicom_date(value: str):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError:
        return None


def _subject_from_path(path: Path):
    for part in path.parts:
        if part.startswith("QIN-BREAST-02-"):
            return part
    return ""


def _role_counts(series_items):
    counts = {}
    for item in series_items:
        role = item["candidate_role"] or "unknown"
        counts[role] = counts.get(role, 0) + 1
    return counts
