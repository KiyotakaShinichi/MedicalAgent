"""Top-level orchestration and file emission for complete synthetic datasets."""

import json
import random
from pathlib import Path

from backend.config import DATA_DIR
from backend.models import Patient
from backend.services.complete_synthetic_dataset_components.constants import (
    COMPLETE_SYNTHETIC_PREFIX,
    COMPLETE_SYNTHETIC_SOURCE,
)
from backend.services.complete_synthetic_dataset_components.journey import (
    _build_patient_journey,
)
from backend.services.complete_synthetic_dataset_components.persistence import (
    _write_journey_to_db,
)
from backend.services.complete_synthetic_dataset_components.profiles import (
    _balanced_response_band,
)
from backend.services.complete_synthetic_dataset_io import (
    data_dictionary as _data_dictionary,
    empty_tables as _empty_tables,
    write_tables as _write_tables,
)


def generate_complete_synthetic_breast_dataset(
    db,
    count=60,
    seed=2027,
    cycles=6,
    output_dir="Data/complete_synthetic_breast_journeys",
    write_db=True,
    patient_prefix=COMPLETE_SYNTHETIC_PREFIX,
    balanced_outcomes=True,
    balanced_subgroups=True,
    missing_rate=0.04,
    noise_level=0.03,
    realism_profile="balanced",
    toxicity_profile="default",
    missingness_mode="mcar",
):
    rng = random.Random(seed)
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = DATA_DIR.parent / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    tables = _empty_tables()
    created = {
        "patients_created": 0,
        "patients_skipped": 0,
        "cycles_per_patient": cycles,
        "patient_prefix": patient_prefix,
        "source": COMPLETE_SYNTHETIC_SOURCE,
        "output_dir": str(output_path),
    }

    for index in range(1, count + 1):
        patient_id = f"{patient_prefix}{index:04d}"
        if write_db and db.query(Patient).filter(Patient.id == patient_id).first():
            created["patients_skipped"] += 1
            continue

        forced_response_band = (
            _balanced_response_band(
                index + ((index - 1) // 10 if balanced_subgroups else 0)
            )
            if balanced_outcomes
            else None
        )
        journey = _build_patient_journey(
            patient_id=patient_id,
            index=index,
            cycles=cycles,
            rng=rng,
            forced_response_band=forced_response_band,
            balanced_subgroups=balanced_subgroups,
            missing_rate=missing_rate,
            noise_level=noise_level,
            realism_profile=realism_profile,
            toxicity_profile=toxicity_profile,
            missingness_mode=missingness_mode,
        )
        for table_name, rows in journey.items():
            tables[table_name].extend(rows)
        created["patients_created"] += 1

        if write_db:
            _write_journey_to_db(db, journey)

    if write_db:
        db.commit()

    file_manifest = _write_tables(output_path, tables)
    summary = {
        **created,
        "table_counts": {name: len(rows) for name, rows in tables.items()},
        "files": file_manifest,
        "generation_options": {
            "seed": seed,
            "cycles": cycles,
            "count_requested": count,
            "balanced_outcomes": balanced_outcomes,
            "balanced_subgroups": balanced_subgroups,
            "missing_rate": missing_rate,
            "noise_level": noise_level,
            "realism_profile": realism_profile,
            "toxicity_profile": toxicity_profile,
            "missingness_mode": missingness_mode,
            "schema_version": "complete_synthetic_breast_journey_v2",
        },
        "warning": (
            "Fully synthetic data for engineering and ML practice only. "
            "It is not clinical evidence and must not be used for patient care."
        ),
    }
    (output_path / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_path / "data_dictionary.json").write_text(
        json.dumps(_data_dictionary(), indent=2),
        encoding="utf-8",
    )
    return summary
