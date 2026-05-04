from sqlalchemy import Column, Date, DateTime, Float, Integer, String, Text, ForeignKey
from sqlalchemy.sql import func

from backend.database import Base


class Patient(Base):
    __tablename__ = "patients"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    diagnosis = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class UserAccount(Base):
    __tablename__ = "user_accounts"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, nullable=False, unique=True, index=True)
    role = Column(String, nullable=False)
    patient_id = Column(String, ForeignKey("patients.id"), nullable=True, index=True)
    display_name = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AccessSession(Base):
    __tablename__ = "access_sessions"

    id = Column(Integer, primary_key=True, index=True)
    token = Column(String, nullable=False, unique=True, index=True)
    role = Column(String, nullable=False)
    patient_id = Column(String, ForeignKey("patients.id"), nullable=True, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class BreastCancerProfile(Base):
    __tablename__ = "breast_cancer_profiles"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, ForeignKey("patients.id"), unique=True, index=True)
    cancer_stage = Column(String, nullable=True)
    er_status = Column(String, nullable=True)
    pr_status = Column(String, nullable=True)
    her2_status = Column(String, nullable=True)
    molecular_subtype = Column(String, nullable=True)
    treatment_intent = Column(String, nullable=True)
    menopausal_status = Column(String, nullable=True)


class LabResult(Base):
    __tablename__ = "lab_results"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, ForeignKey("patients.id"), index=True)
    date = Column(Date, nullable=False)
    wbc = Column(Float, nullable=False)
    hemoglobin = Column(Float, nullable=False)
    platelets = Column(Float, nullable=False)
    source = Column(String, nullable=False, default="manual")
    source_note = Column(Text, nullable=True)


class SymptomReport(Base):
    __tablename__ = "symptom_reports"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, ForeignKey("patients.id"), index=True)
    date = Column(Date, nullable=False)
    symptom = Column(String, nullable=False)
    severity = Column(Integer, nullable=False)
    notes = Column(Text, nullable=True)


class Treatment(Base):
    __tablename__ = "treatments"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, ForeignKey("patients.id"), index=True)
    date = Column(Date, nullable=False)
    cycle = Column(Integer, nullable=False)
    drug = Column(String, nullable=False)


class MedicationLog(Base):
    __tablename__ = "medication_logs"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, ForeignKey("patients.id"), index=True)
    date = Column(Date, nullable=False)
    medication = Column(String, nullable=False)
    dose = Column(String, nullable=True)
    frequency = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    source = Column(String, nullable=False, default="chat_agent")
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, ForeignKey("patients.id"), index=True)
    role = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    intent = Column(String, nullable=True)
    saved_actions_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class CTReport(Base):
    __tablename__ = "ct_reports"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, ForeignKey("patients.id"), index=True)
    date = Column(Date, nullable=False)
    report_type = Column(String, nullable=False)
    findings = Column(Text, nullable=False)
    impression = Column(Text, nullable=False)


class ImagingReport(Base):
    __tablename__ = "imaging_reports"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, ForeignKey("patients.id"), index=True)
    date = Column(Date, nullable=False)
    modality = Column(String, nullable=False)
    report_type = Column(String, nullable=False)
    body_site = Column(String, nullable=True)
    findings = Column(Text, nullable=False)
    impression = Column(Text, nullable=False)


class MRIFileRegistry(Base):
    __tablename__ = "mri_file_registry"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, ForeignKey("patients.id"), index=True)
    scan_date = Column(Date, nullable=True)
    modality = Column(String, nullable=False, default="Breast MRI")
    series_description = Column(String, nullable=True)
    local_path = Column(Text, nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class MRISeriesIndex(Base):
    __tablename__ = "mri_series_index"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, ForeignKey("patients.id"), index=True)
    study_date = Column(Date, nullable=True)
    modality = Column(String, nullable=True)
    series_description = Column(String, nullable=True)
    series_uid = Column(String, nullable=False, unique=True, index=True)
    folder = Column(Text, nullable=False)
    instance_count = Column(Integer, nullable=False, default=0)
    candidate_role = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class PatientReport(Base):
    __tablename__ = "patient_reports"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, ForeignKey("patients.id"), index=True)
    generated_at = Column(DateTime(timezone=True), server_default=func.now())
    report_json = Column(Text, nullable=False)


class PatientUpload(Base):
    __tablename__ = "patient_uploads"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, ForeignKey("patients.id"), index=True)
    upload_type = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    content_type = Column(String, nullable=True)
    local_path = Column(Text, nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, nullable=False, index=True)
    model_version = Column(String, nullable=False, index=True)
    task = Column(String, nullable=False)
    artifact_path = Column(Text, nullable=False)
    metrics_path = Column(Text, nullable=True)
    training_data_path = Column(Text, nullable=True)
    model_metadata_json = Column(Text, nullable=True)
    status = Column(String, nullable=False, default="active")
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class PredictionAuditLog(Base):
    __tablename__ = "prediction_audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, ForeignKey("patients.id"), index=True)
    model_name = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    input_reference = Column(Text, nullable=True)
    prediction_json = Column(Text, nullable=False)
    explanation_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
