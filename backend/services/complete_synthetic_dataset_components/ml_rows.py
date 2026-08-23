"""Cycle-level ML rows, engineered labels, missingness, and outcomes."""

from datetime import timedelta

from backend.services.complete_synthetic_dataset_components.constants import (
    COMPLETE_SYNTHETIC_SOURCE,
)


def _ml_row(
    patient_id,
    profile,
    cycle,
    treatment_date,
    pre_lab,
    nadir,
    recovery,
    mri_size,
    baseline_size,
    symptoms,
    interventions,
    dose_delayed,
    dose_reduced,
    response_strength,
    imaging_rows=None,
):
    max_symptom = max([row["severity"] for row in symptoms], default=0)
    patient_imaging = [
        row
        for row in (imaging_rows or [])
        if row.get("patient_id") == patient_id
    ]
    modalities = {
        str(row.get("modality") or "").lower() for row in patient_imaging
    }
    return {
        "patient_id": patient_id,
        "cycle": cycle,
        "treatment_date": treatment_date,
        "age": profile["age"],
        "stage": profile["stage"],
        "molecular_subtype": profile["subtype"],
        "regimen": profile["regimen"],
        "pre_wbc": pre_lab["wbc"],
        "pre_anc": pre_lab["anc"],
        "pre_hemoglobin": pre_lab["hemoglobin"],
        "pre_platelets": pre_lab["platelets"],
        "nadir_wbc": nadir["wbc"],
        "nadir_anc": nadir["anc"],
        "nadir_hemoglobin": nadir["hemoglobin"],
        "nadir_platelets": nadir["platelets"],
        "recovery_wbc": recovery["wbc"],
        "recovery_hemoglobin": recovery["hemoglobin"],
        "recovery_platelets": recovery["platelets"],
        "mri_tumor_size_cm": mri_size,
        "mri_percent_change_from_baseline": round(
            ((mri_size - baseline_size) / baseline_size) * 100, 2
        ),
        "has_mri": int(any("mri" in modality for modality in modalities)),
        "has_ct": int(any("ct" in modality for modality in modalities)),
        "has_ultrasound": int(
            any("ultrasound" in modality for modality in modalities)
        ),
        "imaging_modality_count": len({m for m in modalities if m}),
        "response_score_percent": round(
            -((mri_size - baseline_size) / baseline_size) * 100, 2
        ),
        "max_symptom_severity": max_symptom,
        "symptom_count": len(symptoms),
        "intervention_count": len(interventions),
        "dose_delayed": int(dose_delayed),
        "dose_reduced": int(dose_reduced),
        "latent_response_strength": round(response_strength, 4),
    }


def _final_outcome(
    patient_id,
    start_date,
    cycles,
    final_size,
    baseline_size,
    response_strength,
    profile,
    rng,
):
    assessment_date = start_date + timedelta(days=cycles * 21 + 55)
    percent_change = ((final_size - baseline_size) / baseline_size) * 100
    if profile["stage"] == "IV":
        response_category = (
            "disease_control" if percent_change < 10 else "progressive_disease"
        )
        cancer_status = (
            "maintenance_systemic_therapy"
            if response_category == "disease_control"
            else "active_disease_needs_review"
        )
        maintenance_plan = "ongoing systemic maintenance and oncology follow-up"
        risk = "high"
    elif response_strength >= 0.78 and (
        final_size <= 0.8 or percent_change <= -78
    ):
        response_category = "complete_response_signal"
        cancer_status = "no_evidence_of_disease"
        maintenance_plan = (
            "routine surveillance"
            if not profile["subtype"].startswith("HR+")
            else "endocrine maintenance plus surveillance"
        )
        risk = rng.choice(["low", "intermediate"])
    elif percent_change <= -45:
        response_category = "partial_response"
        cancer_status = "minimal_residual_disease"
        maintenance_plan = (
            "surgery/radiation planning with maintenance therapy as appropriate"
        )
        risk = "intermediate"
    elif percent_change <= 10:
        response_category = "stable_disease"
        cancer_status = "residual_disease_requires_continued_treatment"
        maintenance_plan = "continued oncology review and possible regimen adjustment"
        risk = "intermediate_high"
    else:
        response_category = "progressive_disease"
        cancer_status = "active_disease_needs_review"
        maintenance_plan = "urgent oncology review for next-line treatment planning"
        risk = "high"

    return {
        "patient_id": patient_id,
        "assessment_date": assessment_date,
        "response_category": response_category,
        "final_tumor_size_cm": round(final_size, 2),
        "percent_change_from_baseline": round(percent_change, 2),
        "cancer_status": cancer_status,
        "maintenance_plan": maintenance_plan,
        "recurrence_risk_band": risk,
        "notes": "Synthetic final outcome label for ML practice and workflow demos only.",
        "source": COMPLETE_SYNTHETIC_SOURCE,
    }


def _add_engineered_labels(ml_row, nadir, interventions, symptoms):
    max_symptom_severity = max(
        [row["severity"] for row in symptoms], default=0
    )
    severe_support_events = {
        "infection_management",
        "blood_transfusion",
        "platelet_support",
    }
    ml_row["toxicity_risk_binary"] = int(
        nadir["anc"] < 1.1
        or nadir["hemoglobin"] < 8.3
        or nadir["platelets"] < 60
        or max_symptom_severity >= 8
    )
    ml_row["urgent_intervention_needed"] = int(
        any(
            row["intervention_type"] in severe_support_events
            for row in interventions
        )
        or nadir["anc"] < 0.8
        or nadir["hemoglobin"] < 7.5
        or nadir["platelets"] < 35
    )
    ml_row["support_intervention_needed"] = int(
        len(interventions) >= 2
        or ml_row["dose_delayed"] == 1
        or ml_row["dose_reduced"] == 1
        or max_symptom_severity >= 8
        or nadir["anc"] < 1.1
    )
    if ml_row["mri_percent_change_from_baseline"] <= -50:
        trend = "strong_response"
    elif ml_row["mri_percent_change_from_baseline"] <= -25:
        trend = "partial_response"
    elif ml_row["mri_percent_change_from_baseline"] <= 10:
        trend = "stable"
    else:
        trend = "progression"
    ml_row["cycle_response_trend_class"] = trend


def _apply_missingness(row, rng, missing_rate, mode="mcar", cycle=None):
    if missing_rate <= 0:
        return
    optional_columns = [
        "pre_anc",
        "pre_platelets",
        "nadir_anc",
        "nadir_platelets",
        "recovery_platelets",
        "mri_tumor_size_cm",
        "mri_percent_change_from_baseline",
        "max_symptom_severity",
    ]
    for column in optional_columns:
        probability = missing_rate
        if mode == "ehr_like":
            # Simulate non-random clinical documentation: ANC/platelets are more
            # likely missing in sparse uploads, while later-cycle symptom/MRI rows
            # are usually more complete after monitoring is established.
            if "anc" in column:
                probability *= 1.8
            elif "platelets" in column:
                probability *= 1.25
            elif column.startswith("mri_"):
                probability *= 0.65 if (cycle or 1) >= 2 else 1.35
            elif column == "max_symptom_severity":
                probability *= 0.75 if (cycle or 1) >= 3 else 1.2
        if rng.random() < min(0.45, probability):
            row[column] = None
