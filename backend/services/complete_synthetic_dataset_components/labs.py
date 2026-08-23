"""CBC generation, treatment-delay rules, and lab-row serialization."""

from backend.services.complete_synthetic_dataset_components.constants import (
    COMPLETE_SYNTHETIC_SOURCE,
)


def _lab_values(wbc, hemoglobin, platelets, rng, noise_level=0.03):
    wbc = _jitter(wbc, rng, noise_level)
    hemoglobin = _jitter(hemoglobin, rng, noise_level)
    platelets = _jitter(platelets, rng, noise_level)
    rbc = max(2.4, hemoglobin / rng.uniform(3.0, 3.35))
    anc = max(0.2, wbc * rng.uniform(0.42, 0.72))
    return {
        "wbc": round(wbc, 2),
        "anc": round(anc, 2),
        "rbc": round(rbc, 2),
        "hemoglobin": round(hemoglobin, 2),
        "platelets": round(platelets, 0),
    }


def _cycle_nadir(
    pre_lab,
    cycle,
    response_strength,
    rng,
    noise_level=0.03,
    toxicity_profile="default",
):
    toxicity = rng.uniform(0.8, 1.35) + (cycle * 0.05)
    if response_strength > 0.7:
        toxicity += 0.10
    if toxicity_profile == "realistic":
        toxicity = rng.uniform(0.58, 1.05) + (cycle * 0.035)
        if response_strength > 0.7:
            toxicity += 0.06
        wbc = max(1.0, pre_lab["wbc"] - rng.uniform(0.7, 2.75) * toxicity)
        hgb = max(8.2, pre_lab["hemoglobin"] - rng.uniform(0.12, 0.82) * toxicity)
        platelets = max(
            70, pre_lab["platelets"] - rng.uniform(12, 82) * toxicity
        )
    else:
        wbc = max(0.6, pre_lab["wbc"] - rng.uniform(1.5, 3.8) * toxicity)
        hgb = max(6.8, pre_lab["hemoglobin"] - rng.uniform(0.35, 1.25) * toxicity)
        platelets = max(
            25, pre_lab["platelets"] - rng.uniform(30, 120) * toxicity
        )
    return _lab_values(wbc, hgb, platelets, rng, noise_level=noise_level)


def _recovery_values(
    nadir,
    baseline_wbc,
    baseline_hgb,
    baseline_platelets,
    rng,
    noise_level=0.03,
):
    wbc = min(baseline_wbc + 0.4, nadir["wbc"] + rng.uniform(1.0, 2.9))
    hgb = min(
        baseline_hgb + 0.2, nadir["hemoglobin"] + rng.uniform(0.15, 0.75)
    )
    platelets = min(
        baseline_platelets + 25, nadir["platelets"] + rng.uniform(35, 110)
    )
    return _lab_values(wbc, hgb, platelets, rng, noise_level=noise_level)


def _needs_delay(nadir):
    return (
        nadir["anc"] < 0.9
        or nadir["wbc"] < 1.4
        or nadir["platelets"] < 55
        or nadir["hemoglobin"] < 7.6
    )


def _needs_reduction(nadir, cycle):
    return cycle >= 2 and (
        nadir["anc"] < 0.75
        or nadir["platelets"] < 45
        or nadir["hemoglobin"] < 7.2
    )


def _lab_row(patient_id, lab_date, cycle, lab_timepoint, lab, note):
    return {
        "patient_id": patient_id,
        "date": lab_date,
        "cycle": cycle,
        "lab_timepoint": lab_timepoint,
        "wbc": lab["wbc"],
        "anc": lab["anc"],
        "rbc": lab["rbc"],
        "hemoglobin": lab["hemoglobin"],
        "platelets": lab["platelets"],
        "source": COMPLETE_SYNTHETIC_SOURCE,
        "note": note,
    }


def _jitter(value, rng, noise_level):
    if noise_level <= 0:
        return value
    return value * (1 + rng.uniform(-noise_level, noise_level))
