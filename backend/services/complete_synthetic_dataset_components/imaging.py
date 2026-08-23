"""Synthetic MRI and optional cross-sectional imaging events."""

from datetime import timedelta

from backend.services.complete_synthetic_dataset_components.constants import (
    COMPLETE_SYNTHETIC_SOURCE,
)


def _next_mri_size(
    current_size,
    baseline_size,
    response_strength,
    cycle,
    cycles,
    rng,
):
    if response_strength < 0.25 and cycle > cycles / 2:
        return round(min(20.0, current_size * rng.uniform(1.02, 1.18)), 2)
    cycle_fraction = cycle / cycles
    target_reduction = response_strength * cycle_fraction
    expected_size = baseline_size * max(0.02, 1 - target_reduction)
    return round(
        min(20.0, max(0.0, expected_size + rng.uniform(-0.18, 0.18))), 2
    )


def _mri_response_text(current_size, baseline_size, response_strength):
    change = (current_size - baseline_size) / baseline_size
    if response_strength >= 0.78 and current_size <= 0.35:
        return "near complete imaging response with minimal residual enhancement"
    if change <= -0.5:
        return "marked interval decrease in size and enhancement"
    if change <= -0.25:
        return "partial interval decrease in tumor size"
    if change >= 0.10:
        return "interval increase in tumor size concerning for progression"
    return "overall stable residual enhancing disease"


def _add_mri_row(
    tables,
    patient_id,
    mri_date,
    timepoint,
    cycle,
    size_cm,
    baseline_size_cm,
    response_text,
    profile,
):
    percent_change = round(
        ((size_cm - baseline_size_cm) / baseline_size_cm) * 100, 1
    )
    tables["mri_reports"].append(
        {
            "patient_id": patient_id,
            "date": mri_date,
            "cycle": cycle,
            "timepoint": timepoint,
            "modality": "Breast MRI",
            "body_site": "Breast",
            "breast_side": profile["breast_side"],
            "location": profile["location"],
            "tumor_size_cm": size_cm,
            "percent_change_from_baseline": percent_change,
            "response_text": response_text,
            "bi_rads": 6 if cycle == 0 else None,
            "source": COMPLETE_SYNTHETIC_SOURCE,
        }
    )


def _add_optional_cross_sectional_imaging(
    tables,
    patient_id,
    imaging_date,
    cycle,
    profile,
    response_strength,
    current_size,
    baseline_size,
    rng,
):
    """Add optional CT and ultrasound report events.

    Patients do not need every modality. This mirrors real monitoring better:
    some have MRI only, some have ultrasound follow-up, some have CT/PET/CT
    when metastatic or abdominal/chest concerns are being evaluated.
    """
    stage_iv = profile.get("stage") == "IV"
    high_concern = stage_iv or response_strength < 0.28
    percent_change = round(
        ((current_size - baseline_size) / baseline_size) * 100, 1
    )

    if rng.random() < (0.45 if high_concern else 0.12):
        metastatic_phrase = (
            "small-volume ascites and indeterminate liver lesions are described; oncology correlation recommended"
            if high_concern and rng.random() < 0.55
            else "no definite metastatic disease is described in this synthetic report"
        )
        tables["mri_reports"].append(
            {
                "patient_id": patient_id,
                "date": imaging_date,
                "cycle": cycle,
                "timepoint": f"cycle_{cycle}_ct",
                "modality": "CT chest/abdomen/pelvis",
                "body_site": "Chest/abdomen/pelvis",
                "breast_side": None,
                "location": "systemic staging",
                "tumor_size_cm": None,
                "percent_change_from_baseline": percent_change,
                "response_text": metastatic_phrase,
                "bi_rads": None,
                "source": COMPLETE_SYNTHETIC_SOURCE,
            }
        )

    if rng.random() < 0.38:
        direction = (
            "decreased"
            if percent_change <= -20
            else "stable"
            if percent_change <= 10
            else "increased"
        )
        tables["mri_reports"].append(
            {
                "patient_id": patient_id,
                "date": imaging_date + timedelta(days=rng.randint(0, 3)),
                "cycle": cycle,
                "timepoint": f"cycle_{cycle}_ultrasound",
                "modality": "Breast ultrasound",
                "body_site": "Breast/axilla",
                "breast_side": profile["breast_side"],
                "location": profile["location"],
                "tumor_size_cm": current_size if rng.random() < 0.75 else None,
                "percent_change_from_baseline": percent_change,
                "response_text": (
                    f"target breast lesion appears {direction} compared with prior "
                    "synthetic measurement"
                ),
                "bi_rads": None,
                "source": COMPLETE_SYNTHETIC_SOURCE,
            }
        )
