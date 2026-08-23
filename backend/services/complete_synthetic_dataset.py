"""Compatibility facade for the complete synthetic breast dataset generator.

Implementation is decomposed by responsibility under
``complete_synthetic_dataset_components``. Existing imports, including the
historical underscore-prefixed helpers, remain available from this module.
"""

# ruff: noqa: F401

# These imports intentionally preserve the module's previous public/de-facto-public
# namespace in addition to the generator and helper compatibility surface.
import json  # noqa: F401
import random  # noqa: F401
from datetime import date, timedelta  # noqa: F401
from pathlib import Path  # noqa: F401

from backend.config import DATA_DIR  # noqa: F401
from backend.models import (  # noqa: F401
    BreastCancerProfile,
    ClinicalIntervention,
    ImagingReport,
    LabResult,
    MedicationLog,
    Patient,
    SymptomReport,
    Treatment,
    TreatmentOutcome,
)
from backend.services.complete_synthetic_dataset_components.constants import (
    COMPLETE_SYNTHETIC_PREFIX,
    COMPLETE_SYNTHETIC_SOURCE,
)
from backend.services.complete_synthetic_dataset_components.generation import (
    generate_complete_synthetic_breast_dataset as _generate_complete_synthetic_breast_dataset,
)
from backend.services.complete_synthetic_dataset_components.imaging import (
    _add_mri_row,
    _add_optional_cross_sectional_imaging,
    _mri_response_text,
    _next_mri_size,
)
from backend.services.complete_synthetic_dataset_components.journey import (
    _build_patient_journey,
)
from backend.services.complete_synthetic_dataset_components.labs import (
    _cycle_nadir,
    _jitter,
    _lab_row,
    _lab_values,
    _needs_delay,
    _needs_reduction,
    _recovery_values,
)
from backend.services.complete_synthetic_dataset_components.ml_rows import (
    _add_engineered_labels,
    _apply_missingness,
    _final_outcome,
    _ml_row,
)
from backend.services.complete_synthetic_dataset_components.persistence import (
    _write_journey_to_db,
)
from backend.services.complete_synthetic_dataset_components.profiles import (
    _balanced_response_band,
    _response_strength,
    _sample_profile,
)
from backend.services.complete_synthetic_dataset_components.support_events import (
    _intervention,
    _interventions_for_cycle,
    _session_medications,
    _symptoms_for_cycle,
)
from backend.services.complete_synthetic_dataset_io import (  # noqa: F401
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
    return _generate_complete_synthetic_breast_dataset(
        db=db,
        count=count,
        seed=seed,
        cycles=cycles,
        output_dir=output_dir,
        write_db=write_db,
        patient_prefix=patient_prefix,
        balanced_outcomes=balanced_outcomes,
        balanced_subgroups=balanced_subgroups,
        missing_rate=missing_rate,
        noise_level=noise_level,
        realism_profile=realism_profile,
        toxicity_profile=toxicity_profile,
        missingness_mode=missingness_mode,
    )
