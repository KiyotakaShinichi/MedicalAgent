import json
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from backend.config import DATA_DIR
from backend.models import (
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


COMPLETE_SYNTHETIC_PREFIX = "COMP-BRCA-"
COMPLETE_SYNTHETIC_SOURCE = "synthetic_complete_breast_journey"


def generate_complete_synthetic_breast_dataset(
    db,
    count=60,
    seed=2027,
    cycles=6,
    output_dir="Data/complete_synthetic_breast_journeys",
    write_db=True,
    patient_prefix=COMPLETE_SYNTHETIC_PREFIX,
    balanced_outcomes=True,
    missing_rate=0.04,
    noise_level=0.03,
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

        forced_response_band = _balanced_response_band(index) if balanced_outcomes else None
        journey = _build_patient_journey(
            patient_id=patient_id,
            index=index,
            cycles=cycles,
            rng=rng,
            forced_response_band=forced_response_band,
            missing_rate=missing_rate,
            noise_level=noise_level,
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
            "balanced_outcomes": balanced_outcomes,
            "missing_rate": missing_rate,
            "noise_level": noise_level,
        },
        "warning": (
            "Fully synthetic data for engineering and ML practice only. "
            "It is not clinical evidence and must not be used for patient care."
        ),
    }
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_path / "data_dictionary.json").write_text(
        json.dumps(_data_dictionary(), indent=2),
        encoding="utf-8",
    )
    return summary


def _empty_tables():
    return {
        "patients": [],
        "diagnoses": [],
        "treatment_sessions": [],
        "labs": [],
        "medications": [],
        "symptoms": [],
        "mri_reports": [],
        "interventions": [],
        "outcomes": [],
        "temporal_ml_rows": [],
    }


def _build_patient_journey(patient_id, index, cycles, rng, forced_response_band=None, missing_rate=0.04, noise_level=0.03):
    profile = _sample_profile(index, rng)
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
    response_strength = _response_strength(profile, rng, forced_response_band=forced_response_band)

    for cycle in range(1, cycles + 1):
        planned_date = start_date + timedelta(days=(cycle - 1) * 21 + cumulative_delay)
        pre_lab_date = planned_date - timedelta(days=1)
        pre_wbc = max(2.4, previous_recovery["wbc"] + rng.uniform(-0.3, 0.45))
        pre_hgb = max(8.6, previous_recovery["hgb"] + rng.uniform(-0.15, 0.25))
        pre_platelets = max(80, previous_recovery["platelets"] + rng.uniform(-12, 25))
        pre_lab = _lab_values(pre_wbc, pre_hgb, pre_platelets, rng, noise_level=noise_level)
        tables["labs"].append(_lab_row(patient_id, pre_lab_date, cycle, "pre_cycle", pre_lab, "Pre-cycle CBC"))

        nadir = _cycle_nadir(pre_lab, cycle, response_strength, rng, noise_level=noise_level)
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

        for medication in _session_medications(patient_id, cycle, actual_date, profile, rng):
            tables["medications"].append(medication)

        tables["labs"].append(_lab_row(
            patient_id,
            actual_date + timedelta(days=9),
            cycle,
            "post_cycle_nadir",
            nadir,
            "Post-cycle CBC nadir",
        ))

        interventions = _interventions_for_cycle(patient_id, cycle, actual_date, nadir, rng)
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

        symptoms = _symptoms_for_cycle(patient_id, cycle, actual_date, nadir, dose_delayed, rng)
        tables["symptoms"].extend(symptoms)

        recovery = _recovery_values(nadir, baseline_wbc, baseline_hgb, baseline_platelets, rng, noise_level=noise_level)
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

        current_size = _next_mri_size(current_size, baseline_size, response_strength, cycle, cycles, rng)
        _add_mri_row(
            tables=tables,
            patient_id=patient_id,
            mri_date=actual_date + timedelta(days=13),
            timepoint=f"cycle_{cycle}",
            cycle=cycle,
            size_cm=current_size,
            baseline_size_cm=baseline_size,
            response_text=_mri_response_text(current_size, baseline_size, response_strength),
            profile=profile,
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
        )
        _add_engineered_labels(ml_row, nadir, interventions, symptoms)
        _apply_missingness(ml_row, rng, missing_rate)
        tables["temporal_ml_rows"].append(ml_row)

    outcome = _final_outcome(patient_id, start_date, cycles, current_size, baseline_size, response_strength, profile, rng)
    tables["outcomes"].append(outcome)
    for row in tables["temporal_ml_rows"]:
        row["final_response_category"] = outcome["response_category"]
        row["final_cancer_status"] = outcome["cancer_status"]
        row["treatment_success_binary"] = 1 if outcome["cancer_status"] in {"no_evidence_of_disease", "minimal_residual_disease"} else 0
        row["maintenance_needed"] = 1 if outcome["maintenance_plan"] != "routine surveillance" else 0
        row["final_response_multiclass"] = outcome["response_category"]

    return tables


def _balanced_response_band(index):
    bands = ["complete", "partial", "stable", "progressive", "maintenance"]
    return bands[(index - 1) % len(bands)]


def _sample_profile(index, rng):
    subtype = rng.choices(
        ["HR+/HER2-", "HER2+", "triple-negative", "HR+/HER2+"],
        weights=[0.45, 0.22, 0.23, 0.10],
        k=1,
    )[0]
    stages = ["IIA", "IIB", "IIIA", "IIIB", "IV"]
    stage = rng.choices(stages, weights=[0.25, 0.30, 0.25, 0.15, 0.05], k=1)[0]
    receptor_map = {
        "HR+/HER2-": ("Positive", "Positive", "Not amplified"),
        "HER2+": ("Negative", "Negative", "Amplified"),
        "triple-negative": ("Negative", "Negative", "Not amplified"),
        "HR+/HER2+": ("Positive", "Positive", "Amplified"),
    }
    regimen_map = {
        "HR+/HER2-": ("dose-dense AC then paclitaxel", ["doxorubicin", "cyclophosphamide", "paclitaxel"]),
        "HER2+": ("TCHP", ["docetaxel", "carboplatin", "trastuzumab", "pertuzumab"]),
        "triple-negative": ("paclitaxel + carboplatin then AC", ["paclitaxel", "carboplatin", "doxorubicin", "cyclophosphamide"]),
        "HR+/HER2+": ("TCHP then endocrine therapy", ["docetaxel", "carboplatin", "trastuzumab", "pertuzumab"]),
    }
    regimen, drugs = regimen_map[subtype]
    er, pr, her2 = receptor_map[subtype]
    return {
        "age": rng.randint(31, 74),
        "stage": stage,
        "er_status": er,
        "pr_status": pr,
        "her2_status": her2,
        "subtype": subtype,
        "grade": rng.choice([2, 3, 3]),
        "menopausal_status": rng.choice(["premenopausal", "postmenopausal", "perimenopausal"]),
        "baseline_tumor_size_cm": round(rng.uniform(2.1, 7.2), 1),
        "nodal_status": rng.choice(["N0", "N1", "N1", "N2", "N3"]),
        "treatment_intent": "palliative disease control" if stage == "IV" else "neoadjuvant curative-intent therapy",
        "regimen": regimen,
        "drugs": drugs,
        "breast_side": rng.choice(["left", "right"]),
        "location": rng.choice(["upper outer quadrant", "upper inner quadrant", "central breast", "lower outer quadrant"]),
    }


def _response_strength(profile, rng, forced_response_band=None):
    if forced_response_band == "complete":
        return rng.uniform(0.82, 0.94)
    if forced_response_band == "partial":
        return rng.uniform(0.56, 0.74)
    if forced_response_band == "stable":
        return rng.uniform(0.28, 0.46)
    if forced_response_band == "progressive":
        return rng.uniform(0.08, 0.22)
    if forced_response_band == "maintenance":
        return rng.uniform(0.42, 0.62)

    base = {
        "HER2+": 0.78,
        "triple-negative": 0.68,
        "HR+/HER2+": 0.72,
        "HR+/HER2-": 0.52,
    }[profile["subtype"]]
    if profile["stage"] in {"IIIB", "IV"}:
        base -= 0.12
    return max(0.1, min(0.95, base + rng.uniform(-0.22, 0.18)))


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


def _cycle_nadir(pre_lab, cycle, response_strength, rng, noise_level=0.03):
    toxicity = rng.uniform(0.8, 1.35) + (cycle * 0.05)
    if response_strength > 0.7:
        toxicity += 0.10
    wbc = max(0.6, pre_lab["wbc"] - rng.uniform(1.5, 3.8) * toxicity)
    hgb = max(6.8, pre_lab["hemoglobin"] - rng.uniform(0.35, 1.25) * toxicity)
    platelets = max(25, pre_lab["platelets"] - rng.uniform(30, 120) * toxicity)
    return _lab_values(wbc, hgb, platelets, rng, noise_level=noise_level)


def _recovery_values(nadir, baseline_wbc, baseline_hgb, baseline_platelets, rng, noise_level=0.03):
    wbc = min(baseline_wbc + 0.4, nadir["wbc"] + rng.uniform(1.0, 2.9))
    hgb = min(baseline_hgb + 0.2, nadir["hemoglobin"] + rng.uniform(0.15, 0.75))
    platelets = min(baseline_platelets + 25, nadir["platelets"] + rng.uniform(35, 110))
    return _lab_values(wbc, hgb, platelets, rng, noise_level=noise_level)


def _needs_delay(nadir):
    return nadir["anc"] < 0.9 or nadir["wbc"] < 1.4 or nadir["platelets"] < 55 or nadir["hemoglobin"] < 7.6


def _needs_reduction(nadir, cycle):
    return cycle >= 2 and (nadir["anc"] < 0.75 or nadir["platelets"] < 45 or nadir["hemoglobin"] < 7.2)


def _session_medications(patient_id, cycle, actual_date, profile, rng):
    rows = [
        {
            "patient_id": patient_id,
            "date": actual_date,
            "cycle": cycle,
            "medication": profile["regimen"],
            "dose": "per protocol",
            "frequency": "every 21 days",
            "purpose": "anti-cancer treatment",
            "notes": "Synthetic scheduled systemic therapy session.",
            "source": COMPLETE_SYNTHETIC_SOURCE,
        },
        {
            "patient_id": patient_id,
            "date": actual_date,
            "cycle": cycle,
            "medication": "ondansetron",
            "dose": "8 mg",
            "frequency": "as needed",
            "purpose": "nausea prevention",
            "notes": "Synthetic supportive medication.",
            "source": COMPLETE_SYNTHETIC_SOURCE,
        },
    ]
    if rng.random() < 0.8:
        rows.append({
            "patient_id": patient_id,
            "date": actual_date,
            "cycle": cycle,
            "medication": "dexamethasone",
            "dose": "8 mg",
            "frequency": "daily for 2 days",
            "purpose": "infusion support",
            "notes": "Synthetic supportive medication.",
            "source": COMPLETE_SYNTHETIC_SOURCE,
        })
    return rows


def _interventions_for_cycle(patient_id, cycle, actual_date, nadir, rng):
    rows = []
    if nadir["anc"] < 1.0 or nadir["wbc"] < 1.6:
        rows.append(_intervention(
            patient_id, cycle, actual_date + timedelta(days=10),
            "growth_factor_support",
            "Synthetic severe neutropenia / low WBC support event.",
            "filgrastim" if rng.random() < 0.55 else "pegfilgrastim",
            "per protocol",
        ))
    if nadir["hemoglobin"] < 8.0:
        rows.append(_intervention(
            patient_id, cycle, actual_date + timedelta(days=11),
            "blood_transfusion",
            "Synthetic symptomatic anemia / low hemoglobin support event.",
            "packed red blood cells",
            "1-2 units",
        ))
    if nadir["platelets"] < 50:
        rows.append(_intervention(
            patient_id, cycle, actual_date + timedelta(days=11),
            "platelet_support",
            "Synthetic thrombocytopenia support event.",
            "platelet transfusion",
            "per protocol",
        ))
    if (nadir["anc"] < 0.8 or nadir["wbc"] < 1.2) and rng.random() < 0.45:
        rows.append(_intervention(
            patient_id, cycle, actual_date + timedelta(days=12),
            "infection_management",
            "Synthetic febrile neutropenia / infection concern requiring urgent review.",
            "broad-spectrum antibiotics",
            "per protocol",
        ))
    return rows


def _intervention(patient_id, cycle, event_date, intervention_type, reason, product, dose):
    return {
        "patient_id": patient_id,
        "date": event_date,
        "cycle": cycle,
        "intervention_type": intervention_type,
        "reason": reason,
        "medication_or_product": product,
        "dose": dose,
        "notes": "Synthetic clinical support event for temporal monitoring data.",
        "source": COMPLETE_SYNTHETIC_SOURCE,
    }


def _symptoms_for_cycle(patient_id, cycle, actual_date, nadir, dose_delayed, rng):
    rows = []
    candidates = [
        ("fatigue", min(10, int(3 + (9 - nadir["hemoglobin"]) + rng.randint(0, 3)))),
        ("nausea", rng.randint(2, 7)),
    ]
    if nadir["anc"] < 1.0 or nadir["wbc"] < 1.6:
        candidates.append(("fever", rng.randint(6, 9)))
    if nadir["platelets"] < 70:
        candidates.append(("bruising", rng.randint(4, 8)))
    if cycle >= 3 and rng.random() < 0.35:
        candidates.append(("neuropathy", rng.randint(3, 7)))
    if dose_delayed:
        candidates.append(("anxiety", rng.randint(3, 7)))

    for symptom, severity in rng.sample(candidates, k=min(len(candidates), rng.randint(1, 3))):
        rows.append({
            "patient_id": patient_id,
            "date": actual_date + timedelta(days=rng.randint(4, 13)),
            "cycle": cycle,
            "symptom": symptom,
            "severity": max(1, min(10, severity)),
            "notes": "Synthetic symptom report during treatment cycle.",
            "source": COMPLETE_SYNTHETIC_SOURCE,
        })
    return rows


def _next_mri_size(current_size, baseline_size, response_strength, cycle, cycles, rng):
    if response_strength < 0.25 and cycle > cycles / 2:
        return round(current_size * rng.uniform(1.02, 1.18), 2)
    cycle_fraction = cycle / cycles
    target_reduction = response_strength * cycle_fraction
    expected_size = baseline_size * max(0.02, 1 - target_reduction)
    return round(max(0.0, expected_size + rng.uniform(-0.18, 0.18)), 2)


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


def _add_mri_row(tables, patient_id, mri_date, timepoint, cycle, size_cm, baseline_size_cm, response_text, profile):
    percent_change = round(((size_cm - baseline_size_cm) / baseline_size_cm) * 100, 1)
    tables["mri_reports"].append({
        "patient_id": patient_id,
        "date": mri_date,
        "cycle": cycle,
        "timepoint": timepoint,
        "modality": "Breast MRI",
        "breast_side": profile["breast_side"],
        "location": profile["location"],
        "tumor_size_cm": size_cm,
        "percent_change_from_baseline": percent_change,
        "response_text": response_text,
        "bi_rads": 6 if cycle == 0 else None,
        "source": COMPLETE_SYNTHETIC_SOURCE,
    })


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
):
    max_symptom = max([row["severity"] for row in symptoms], default=0)
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
        "mri_percent_change_from_baseline": round(((mri_size - baseline_size) / baseline_size) * 100, 2),
        "max_symptom_severity": max_symptom,
        "symptom_count": len(symptoms),
        "intervention_count": len(interventions),
        "dose_delayed": int(dose_delayed),
        "dose_reduced": int(dose_reduced),
        "latent_response_strength": round(response_strength, 4),
    }


def _final_outcome(patient_id, start_date, cycles, final_size, baseline_size, response_strength, profile, rng):
    assessment_date = start_date + timedelta(days=cycles * 21 + 55)
    percent_change = ((final_size - baseline_size) / baseline_size) * 100
    if profile["stage"] == "IV":
        response_category = "disease_control" if percent_change < 10 else "progressive_disease"
        cancer_status = "maintenance_systemic_therapy" if response_category == "disease_control" else "active_disease_needs_review"
        maintenance_plan = "ongoing systemic maintenance and oncology follow-up"
        risk = "high"
    elif response_strength >= 0.78 and (final_size <= 0.8 or percent_change <= -78):
        response_category = "complete_response_signal"
        cancer_status = "no_evidence_of_disease"
        maintenance_plan = "routine surveillance" if not profile["subtype"].startswith("HR+") else "endocrine maintenance plus surveillance"
        risk = rng.choice(["low", "intermediate"])
    elif percent_change <= -45:
        response_category = "partial_response"
        cancer_status = "minimal_residual_disease"
        maintenance_plan = "surgery/radiation planning with maintenance therapy as appropriate"
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
    max_symptom_severity = max([row["severity"] for row in symptoms], default=0)
    severe_support_events = {"infection_management", "blood_transfusion", "platelet_support"}
    ml_row["toxicity_risk_binary"] = int(
        nadir["anc"] < 1.1
        or nadir["hemoglobin"] < 8.3
        or nadir["platelets"] < 60
        or max_symptom_severity >= 8
    )
    ml_row["urgent_intervention_needed"] = int(
        any(row["intervention_type"] in severe_support_events for row in interventions)
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


def _apply_missingness(row, rng, missing_rate):
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
        if rng.random() < missing_rate:
            row[column] = None


def _jitter(value, rng, noise_level):
    if noise_level <= 0:
        return value
    return value * (1 + rng.uniform(-noise_level, noise_level))


def _write_journey_to_db(db, journey):
    patient = journey["patients"][0]
    diagnosis = journey["diagnoses"][0]
    db.add(Patient(
        id=patient["patient_id"],
        name=patient["name"],
        diagnosis=patient["diagnosis"],
    ))
    db.add(BreastCancerProfile(
        patient_id=patient["patient_id"],
        cancer_stage=diagnosis["stage"],
        er_status=diagnosis["er_status"],
        pr_status=diagnosis["pr_status"],
        her2_status=diagnosis["her2_status"],
        molecular_subtype=diagnosis["molecular_subtype"],
        treatment_intent=diagnosis["treatment_intent"],
        menopausal_status=diagnosis["menopausal_status"],
    ))
    for row in journey["treatment_sessions"]:
        db.add(Treatment(
            patient_id=row["patient_id"],
            date=row["actual_date"],
            cycle=row["cycle"],
            drug=row["regimen"],
        ))
    for row in journey["labs"]:
        db.add(LabResult(
            patient_id=row["patient_id"],
            date=row["date"],
            wbc=row["wbc"],
            hemoglobin=row["hemoglobin"],
            platelets=row["platelets"],
            source=row["source"],
            source_note=f"{row['lab_timepoint']}: ANC {row['anc']}, RBC {row['rbc']}. {row['note']}",
        ))
    for row in journey["medications"]:
        db.add(MedicationLog(
            patient_id=row["patient_id"],
            date=row["date"],
            medication=row["medication"],
            dose=row["dose"],
            frequency=row["frequency"],
            notes=row["notes"],
            source=row["source"],
        ))
    for row in journey["symptoms"]:
        db.add(SymptomReport(
            patient_id=row["patient_id"],
            date=row["date"],
            symptom=row["symptom"],
            severity=row["severity"],
            notes=row["notes"],
        ))
    for row in journey["mri_reports"]:
        db.add(ImagingReport(
            patient_id=row["patient_id"],
            date=row["date"],
            modality=row["modality"],
            report_type=f"Synthetic {row['timepoint']} MRI",
            body_site="Breast",
            findings=(
                f"Synthetic report: {row['breast_side']} breast {row['location']} enhancing mass "
                f"measuring {row['tumor_size_cm']} cm. {row['response_text']}."
            ),
            impression="Synthetic breast MRI report for software testing only.",
        ))
    for row in journey["interventions"]:
        db.add(ClinicalIntervention(
            patient_id=row["patient_id"],
            date=row["date"],
            intervention_type=row["intervention_type"],
            reason=row["reason"],
            medication_or_product=row["medication_or_product"],
            dose=row["dose"],
            notes=row["notes"],
            source=row["source"],
        ))
    outcome = journey["outcomes"][0]
    db.add(TreatmentOutcome(
        patient_id=outcome["patient_id"],
        assessment_date=outcome["assessment_date"],
        response_category=outcome["response_category"],
        cancer_status=outcome["cancer_status"],
        maintenance_plan=outcome["maintenance_plan"],
        recurrence_risk_band=outcome["recurrence_risk_band"],
        notes=outcome["notes"],
        source=outcome["source"],
    ))


def _write_tables(output_path, tables):
    manifest = {}
    for name, rows in tables.items():
        file_path = output_path / f"{name}.csv"
        pd.DataFrame(rows).to_csv(file_path, index=False)
        manifest[name] = str(file_path)
    return manifest


def _data_dictionary():
    return {
        "patients": "One row per synthetic patient.",
        "diagnoses": "Synthetic diagnosis and receptor/subtype profile.",
        "treatment_sessions": "One row per scheduled treatment cycle with regimen, dates, and dose status.",
        "labs": "CBC rows at baseline/pre-cycle/nadir/recovery, with WBC, ANC, RBC, hemoglobin, and platelets.",
        "medications": "Anti-cancer regimen entries, supportive medications, and intervention medications/products.",
        "symptoms": "Patient-reported symptoms around treatment cycles.",
        "mri_reports": "Synthetic MRI measurements at baseline and every treatment cycle.",
        "interventions": "Clinical support events such as growth-factor support, transfusions, antibiotics, or urgent review.",
        "outcomes": "Synthetic end-of-journey response labels and maintenance status.",
        "temporal_ml_rows": "Training-ready cycle-level features with final outcome labels.",
        "extra_labels": "Synthetic labels include treatment_success_binary, maintenance_needed, toxicity_risk_binary, support_intervention_needed, urgent_intervention_needed, final_response_multiclass, and cycle_response_trend_class.",
        "warning": "All tables are synthetic and should be used only for engineering demos and ML practice.",
    }
