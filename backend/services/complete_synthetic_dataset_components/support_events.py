"""Treatment-session medications, interventions, and symptom events."""

from datetime import timedelta

from backend.services.complete_synthetic_dataset_components.constants import (
    COMPLETE_SYNTHETIC_SOURCE,
)


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
        rows.append(
            {
                "patient_id": patient_id,
                "date": actual_date,
                "cycle": cycle,
                "medication": "dexamethasone",
                "dose": "8 mg",
                "frequency": "daily for 2 days",
                "purpose": "infusion support",
                "notes": "Synthetic supportive medication.",
                "source": COMPLETE_SYNTHETIC_SOURCE,
            }
        )
    return rows


def _interventions_for_cycle(patient_id, cycle, actual_date, nadir, rng):
    rows = []
    if nadir["anc"] < 1.0 or nadir["wbc"] < 1.6:
        rows.append(
            _intervention(
                patient_id,
                cycle,
                actual_date + timedelta(days=10),
                "growth_factor_support",
                "Synthetic severe neutropenia / low WBC support event.",
                "filgrastim" if rng.random() < 0.55 else "pegfilgrastim",
                "per protocol",
            )
        )
    if nadir["hemoglobin"] < 8.0:
        rows.append(
            _intervention(
                patient_id,
                cycle,
                actual_date + timedelta(days=11),
                "blood_transfusion",
                "Synthetic symptomatic anemia / low hemoglobin support event.",
                "packed red blood cells",
                "1-2 units",
            )
        )
    if nadir["platelets"] < 50:
        rows.append(
            _intervention(
                patient_id,
                cycle,
                actual_date + timedelta(days=11),
                "platelet_support",
                "Synthetic thrombocytopenia support event.",
                "platelet transfusion",
                "per protocol",
            )
        )
    if (nadir["anc"] < 0.8 or nadir["wbc"] < 1.2) and rng.random() < 0.45:
        rows.append(
            _intervention(
                patient_id,
                cycle,
                actual_date + timedelta(days=12),
                "infection_management",
                "Synthetic febrile neutropenia / infection concern requiring urgent review.",
                "broad-spectrum antibiotics",
                "per protocol",
            )
        )
    return rows


def _intervention(
    patient_id,
    cycle,
    event_date,
    intervention_type,
    reason,
    product,
    dose,
):
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
        (
            "fatigue",
            min(10, int(3 + (9 - nadir["hemoglobin"]) + rng.randint(0, 3))),
        ),
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

    for symptom, severity in rng.sample(
        candidates, k=min(len(candidates), rng.randint(1, 3))
    ):
        rows.append(
            {
                "patient_id": patient_id,
                "date": actual_date + timedelta(days=rng.randint(4, 13)),
                "cycle": cycle,
                "symptom": symptom,
                "severity": max(1, min(10, severity)),
                "notes": "Synthetic symptom report during treatment cycle.",
                "source": COMPLETE_SYNTHETIC_SOURCE,
            }
        )
    return rows
