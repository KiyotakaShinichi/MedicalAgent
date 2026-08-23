"""Behavioral lock for the decomposed complete synthetic dataset generator."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import random
from datetime import date, datetime

import backend.services.complete_synthetic_dataset as dataset


EXPECTED_SIGNATURES = {
    "generate_complete_synthetic_breast_dataset": "(db, count=60, seed=2027, cycles=6, output_dir='Data/complete_synthetic_breast_journeys', write_db=True, patient_prefix='COMP-BRCA-', balanced_outcomes=True, balanced_subgroups=True, missing_rate=0.04, noise_level=0.03, realism_profile='balanced', toxicity_profile='default', missingness_mode='mcar')",
    "_build_patient_journey": "(patient_id, index, cycles, rng, forced_response_band=None, balanced_subgroups=True, missing_rate=0.04, noise_level=0.03, realism_profile='balanced', toxicity_profile='default', missingness_mode='mcar')",
    "_balanced_response_band": "(index)",
    "_sample_profile": "(index, rng, balanced_subgroups=True, realism_profile='balanced')",
    "_response_strength": "(profile, rng, forced_response_band=None)",
    "_lab_values": "(wbc, hemoglobin, platelets, rng, noise_level=0.03)",
    "_cycle_nadir": "(pre_lab, cycle, response_strength, rng, noise_level=0.03, toxicity_profile='default')",
    "_recovery_values": "(nadir, baseline_wbc, baseline_hgb, baseline_platelets, rng, noise_level=0.03)",
    "_needs_delay": "(nadir)",
    "_needs_reduction": "(nadir, cycle)",
    "_session_medications": "(patient_id, cycle, actual_date, profile, rng)",
    "_interventions_for_cycle": "(patient_id, cycle, actual_date, nadir, rng)",
    "_intervention": "(patient_id, cycle, event_date, intervention_type, reason, product, dose)",
    "_symptoms_for_cycle": "(patient_id, cycle, actual_date, nadir, dose_delayed, rng)",
    "_next_mri_size": "(current_size, baseline_size, response_strength, cycle, cycles, rng)",
    "_mri_response_text": "(current_size, baseline_size, response_strength)",
    "_add_mri_row": "(tables, patient_id, mri_date, timepoint, cycle, size_cm, baseline_size_cm, response_text, profile)",
    "_add_optional_cross_sectional_imaging": "(tables, patient_id, imaging_date, cycle, profile, response_strength, current_size, baseline_size, rng)",
    "_lab_row": "(patient_id, lab_date, cycle, lab_timepoint, lab, note)",
    "_ml_row": "(patient_id, profile, cycle, treatment_date, pre_lab, nadir, recovery, mri_size, baseline_size, symptoms, interventions, dose_delayed, dose_reduced, response_strength, imaging_rows=None)",
    "_final_outcome": "(patient_id, start_date, cycles, final_size, baseline_size, response_strength, profile, rng)",
    "_add_engineered_labels": "(ml_row, nadir, interventions, symptoms)",
    "_apply_missingness": "(row, rng, missing_rate, mode='mcar', cycle=None)",
    "_jitter": "(value, rng, noise_level)",
    "_write_journey_to_db": "(db, journey)",
}

EXPECTED_TABLE_ORDER = [
    "patients",
    "diagnoses",
    "treatment_sessions",
    "labs",
    "medications",
    "symptoms",
    "mri_reports",
    "interventions",
    "outcomes",
    "temporal_ml_rows",
]

EXPECTED_ML_KEY_ORDER = [
    "patient_id",
    "cycle",
    "treatment_date",
    "age",
    "stage",
    "molecular_subtype",
    "regimen",
    "pre_wbc",
    "pre_anc",
    "pre_hemoglobin",
    "pre_platelets",
    "nadir_wbc",
    "nadir_anc",
    "nadir_hemoglobin",
    "nadir_platelets",
    "recovery_wbc",
    "recovery_hemoglobin",
    "recovery_platelets",
    "mri_tumor_size_cm",
    "mri_percent_change_from_baseline",
    "has_mri",
    "has_ct",
    "has_ultrasound",
    "imaging_modality_count",
    "response_score_percent",
    "max_symptom_severity",
    "symptom_count",
    "intervention_count",
    "dose_delayed",
    "dose_reduced",
    "latent_response_strength",
    "toxicity_risk_binary",
    "urgent_intervention_needed",
    "support_intervention_needed",
    "cycle_response_trend_class",
    "final_response_category",
    "final_cancer_status",
    "treatment_success_binary",
    "maintenance_needed",
    "final_response_multiclass",
]

EXPECTED_JOURNEY_SHA256 = (
    "4e8fcf66721bec8cc929446f7b20c2ce0c56fe76c495b3eb048cf30b25b7f805"
)
EXPECTED_DIFFERENT_SEED_SHA256 = (
    "c04689aa1eae6870508bf49c01809ff9e8b8308063a0b341edd388235bc68613"
)
EXPECTED_TEMPORAL_ML_CSV_SHA256 = (
    "92bd518f1b6ef43c7f944b708627f7bdb3072077ebced15cef660526f0baad88"
)


def _canonical(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def _fingerprint(value):
    payload = json.dumps(
        _canonical(value), ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _representative_journey(seed):
    return dataset._build_patient_journey(
        patient_id="TEST-COMP-0001",
        index=1,
        cycles=3,
        rng=random.Random(seed),
        forced_response_band="strong",
        balanced_subgroups=True,
        missing_rate=0.04,
        noise_level=0.03,
        realism_profile="balanced",
        toxicity_profile="default",
        missingness_mode="mcar",
    )


def test_facade_preserves_symbols_signatures_and_defaults():
    assert dataset.COMPLETE_SYNTHETIC_PREFIX == "COMP-BRCA-"
    assert dataset.COMPLETE_SYNTHETIC_SOURCE == "synthetic_complete_breast_journey"
    for name, expected_signature in EXPECTED_SIGNATURES.items():
        assert hasattr(dataset, name)
        assert str(inspect.signature(getattr(dataset, name))) == expected_signature

    signature = inspect.signature(dataset.generate_complete_synthetic_breast_dataset)
    assert signature.parameters["seed"].default == 2027
    assert dataset.generate_complete_synthetic_breast_dataset.__module__ == (
        "backend.services.complete_synthetic_dataset"
    )


def test_seeded_journey_fingerprint_schema_labels_and_timeline_are_unchanged():
    first = _representative_journey(2027)
    repeated = _representative_journey(2027)
    different = _representative_journey(2028)

    assert _fingerprint(first) == EXPECTED_JOURNEY_SHA256
    assert _fingerprint(repeated) == EXPECTED_JOURNEY_SHA256
    assert _fingerprint(different) == EXPECTED_DIFFERENT_SEED_SHA256
    assert _fingerprint(different) != EXPECTED_JOURNEY_SHA256

    assert list(first) == EXPECTED_TABLE_ORDER
    assert list(first["temporal_ml_rows"][0]) == EXPECTED_ML_KEY_ORDER
    assert [row["cycle"] for row in first["treatment_sessions"]] == [1, 2, 3]
    assert [row["cycle"] for row in first["temporal_ml_rows"]] == [1, 2, 3]
    assert [row["actual_date"] for row in first["treatment_sessions"]] == sorted(
        row["actual_date"] for row in first["treatment_sessions"]
    )
    assert first["patients"][0]["patient_id"] == "TEST-COMP-0001"
    assert first["outcomes"][0]["response_category"] == "partial_response"
    assert first["temporal_ml_rows"][0]["treatment_success_binary"] == 1
    assert first["temporal_ml_rows"][0]["final_response_multiclass"] == (
        "partial_response"
    )


def test_public_generator_preserves_counts_ids_and_serialized_ml_rows(tmp_path):
    summary = dataset.generate_complete_synthetic_breast_dataset(
        db=None,
        count=3,
        seed=2027,
        cycles=3,
        output_dir=str(tmp_path),
        write_db=False,
    )

    assert summary["patients_created"] == 3
    assert summary["patients_skipped"] == 0
    assert summary["cycles_per_patient"] == 3
    assert summary["table_counts"] == {
        "patients": 3,
        "diagnoses": 3,
        "treatment_sessions": 9,
        "labs": 27,
        "medications": 35,
        "symptoms": 17,
        "mri_reports": 17,
        "interventions": 8,
        "outcomes": 3,
        "temporal_ml_rows": 9,
    }
    with (tmp_path / "patients.csv").open(encoding="utf-8-sig") as handle:
        patient_ids = [row["patient_id"] for row in csv.DictReader(handle)]
    assert patient_ids == [
        "COMP-BRCA-0001",
        "COMP-BRCA-0002",
        "COMP-BRCA-0003",
    ]

    ml_csv_hash = hashlib.sha256(
        (tmp_path / "temporal_ml_rows.csv").read_bytes()
    ).hexdigest()
    assert ml_csv_hash == EXPECTED_TEMPORAL_ML_CSV_SHA256
