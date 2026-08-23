"""Composition of one longitudinal synthetic breast-cancer journey."""

from datetime import date, timedelta

from backend.services.complete_synthetic_dataset_components.constants import (
    COMPLETE_SYNTHETIC_SOURCE,
)
from backend.services.complete_synthetic_dataset_components.imaging import _add_mri_row, _add_optional_cross_sectional_imaging, _mri_response_text, _next_mri_size
from backend.services.complete_synthetic_dataset_components.labs import _cycle_nadir, _lab_row, _lab_values, _needs_delay, _needs_reduction, _recovery_values
from backend.services.complete_synthetic_dataset_components.ml_rows import _add_engineered_labels, _apply_missingness, _final_outcome, _ml_row
from backend.services.complete_synthetic_dataset_components.profiles import _response_strength, _sample_profile
from backend.services.complete_synthetic_dataset_components.support_events import _interventions_for_cycle, _session_medications, _symptoms_for_cycle
from backend.services.complete_synthetic_dataset_io import empty_tables as _empty_tables


def _build_patient_journey(
    patient_id,
    index,
    cycles,
    rng,
    forced_response_band=None,
    balanced_subgroups=True,
    missing_rate=0.04,
    noise_level=0.03,
    realism_profile="balanced",
    toxicity_profile="default",
    missingness_mode="mcar",
):
    profile = _sample_profile(
        index,
        rng,
        balanced_subgroups=balanced_subgroups,
        realism_profile=realism_profile,
    )
    start_date = date(2025, 1, 6) + timedelta(days=index * 2)
    diagnosis_date = start_date - timedelta(days=rng.randint(14, 45))
    baseline_size = profile["baseline_tumor_size_cm"]
    current_size = baseline_size
    cumulative_delay = 0
    baseline_wbc = rng.uniform(5.1, 8.8)
    baseline_hgb = rng.uniform(11.4, 14.0)
    baseline_platelets = rng.uniform(185, 345)

    tables = _empty_tables()
    tables["patients"].append({
        "patient_id": patient_id,
        "name": f"Complete Synthetic Journey {index:04d}",
        "age": profile["age"],
        "sex": "female",
        "diagnosis": "Doctor-confirmed breast cancer - synthetic complete journey",
        "created_source": COMPLETE_SYNTHETIC_SOURCE,
    })
    tables["diagnoses"].append({
        "patient_id": patient_id,
        "diagnosis_date": diagnosis_date,
        "primary_diagnosis": "Invasive breast carcinoma",
        "stage": profile["stage"],
        "er_status": profile["er_status"],
        "pr_status": profile["pr_status"],
        "her2_status": profile["her2_status"],
        "molecular_subtype": profile["subtype"],
        "grade": profile["grade"],
        "menopausal_status": profile["menopausal_status"],
        "baseline_tumor_size_cm": baseline_size,
        "baseline_nodal_status": profile["nodal_status"],
        "treatment_intent": profile["treatment_intent"],
    })
    _add_mri_row(
        tables=tables,
        patient_id=patient_id,
        mri_date=start_date - timedelta(days=4),
        timepoint="baseline",
        cycle=0,
        size_cm=current_size,
        baseline_size_cm=baseline_size,
        response_text="baseline enhancing breast mass before treatment",
        profile=profile,
    )

    previous_recovery = {
        "wbc": baseline_wbc,
        "hgb": baseline_hgb,
        "platelets": baseline_platelets,
    }
    response_strength = _response_strength(
        profile, rng, forced_response_band=forced_response_band
    )

    for cycle in range(1, cycles + 1):
        planned_date = start_date + timedelta(
            days=(cycle - 1) * 21 + cumulative_delay
        )
        pre_lab_date = planned_date - timedelta(days=1)
        pre_wbc = max(2.4, previous_recovery["wbc"] + rng.uniform(-0.3, 0.45))
        pre_hgb = max(8.6, previous_recovery["hgb"] + rng.uniform(-0.15, 0.25))
        pre_platelets = max(
            80, previous_recovery["platelets"] + rng.uniform(-12, 25)
        )
        pre_lab = _lab_values(
            pre_wbc, pre_hgb, pre_platelets, rng, noise_level=noise_level
        )
        tables["labs"].append(
            _lab_row(
                patient_id,
                pre_lab_date,
                cycle,
                "pre_cycle",
                pre_lab,
                "Pre-cycle CBC",
            )
        )

        nadir = _cycle_nadir(
            pre_lab,
            cycle,
            response_strength,
            rng,
            noise_level=noise_level,
            toxicity_profile=toxicity_profile,
        )
        dose_delayed = _needs_delay(nadir)
        dose_reduced = _needs_reduction(nadir, cycle)
        if dose_delayed:
            cumulative_delay += 7
        actual_date = planned_date + timedelta(days=7 if dose_delayed else 0)

        tables["treatment_sessions"].append({
            "patient_id": patient_id,
            "cycle": cycle,
            "planned_date": planned_date,
            "actual_date": actual_date,
            "regimen": profile["regimen"],
            "drugs": "; ".join(profile["drugs"]),
            "cycle_status": "delayed" if dose_delayed else "given",
            "dose_adjustment": "dose_reduced" if dose_reduced else "standard",
            "intent": profile["treatment_intent"],
            "source": COMPLETE_SYNTHETIC_SOURCE,
        })

        for medication in _session_medications(
            patient_id, cycle, actual_date, profile, rng
        ):
            tables["medications"].append(medication)

        tables["labs"].append(_lab_row(
            patient_id,
            actual_date + timedelta(days=9),
            cycle,
            "post_cycle_nadir",
            nadir,
            "Post-cycle CBC nadir",
        ))

        interventions = _interventions_for_cycle(
            patient_id, cycle, actual_date, nadir, rng
        )
        tables["interventions"].extend(interventions)
        for intervention in interventions:
            if intervention["medication_or_product"]:
                tables["medications"].append({
                    "patient_id": patient_id,
                    "date": intervention["date"],
                    "cycle": cycle,
                    "medication": intervention["medication_or_product"],
                    "dose": intervention["dose"],
                    "frequency": "clinical support event",
                    "purpose": intervention["intervention_type"],
                    "notes": intervention["reason"],
                    "source": COMPLETE_SYNTHETIC_SOURCE,
                })

        symptoms = _symptoms_for_cycle(
            patient_id, cycle, actual_date, nadir, dose_delayed, rng
        )
        tables["symptoms"].extend(symptoms)

        recovery = _recovery_values(
            nadir,
            baseline_wbc,
            baseline_hgb,
            baseline_platelets,
            rng,
            noise_level=noise_level,
        )
        tables["labs"].append(_lab_row(
            patient_id,
            actual_date + timedelta(days=18),
            cycle,
            "recovery",
            recovery,
            "CBC recovery check before next cycle",
        ))
        previous_recovery = {
            "wbc": recovery["wbc"],
            "hgb": recovery["hemoglobin"],
            "platelets": recovery["platelets"],
        }

        current_size = _next_mri_size(
            current_size,
            baseline_size,
            response_strength,
            cycle,
            cycles,
            rng,
        )
        _add_mri_row(
            tables=tables,
            patient_id=patient_id,
            mri_date=actual_date + timedelta(days=13),
            timepoint=f"cycle_{cycle}",
            cycle=cycle,
            size_cm=current_size,
            baseline_size_cm=baseline_size,
            response_text=_mri_response_text(
                current_size, baseline_size, response_strength
            ),
            profile=profile,
        )
        _add_optional_cross_sectional_imaging(
            tables=tables,
            patient_id=patient_id,
            imaging_date=actual_date + timedelta(days=14),
            cycle=cycle,
            profile=profile,
            response_strength=response_strength,
            current_size=current_size,
            baseline_size=baseline_size,
            rng=rng,
        )

        ml_row = _ml_row(
            patient_id=patient_id,
            profile=profile,
            cycle=cycle,
            treatment_date=actual_date,
            pre_lab=pre_lab,
            nadir=nadir,
            recovery=recovery,
            mri_size=current_size,
            baseline_size=baseline_size,
            symptoms=symptoms,
            interventions=interventions,
            dose_delayed=dose_delayed,
            dose_reduced=dose_reduced,
            response_strength=response_strength,
            imaging_rows=tables["mri_reports"],
        )
        _add_engineered_labels(ml_row, nadir, interventions, symptoms)
        _apply_missingness(
            ml_row, rng, missing_rate, mode=missingness_mode, cycle=cycle
        )
        tables["temporal_ml_rows"].append(ml_row)

    outcome = _final_outcome(
        patient_id,
        start_date,
        cycles,
        current_size,
        baseline_size,
        response_strength,
        profile,
        rng,
    )
    tables["outcomes"].append(outcome)
    for row in tables["temporal_ml_rows"]:
        row["final_response_category"] = outcome["response_category"]
        row["final_cancer_status"] = outcome["cancer_status"]
        row["treatment_success_binary"] = (
            1
            if outcome["cancer_status"]
            in {"no_evidence_of_disease", "minimal_residual_disease"}
            else 0
        )
        row["maintenance_needed"] = (
            1 if outcome["maintenance_plan"] != "routine surveillance" else 0
        )
        row["final_response_multiclass"] = outcome["response_category"]

    return tables
