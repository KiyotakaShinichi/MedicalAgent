import random
from datetime import date, timedelta

from backend.models import (
    BreastCancerProfile,
    ImagingReport,
    LabResult,
    MedicationLog,
    Patient,
    SymptomReport,
    Treatment,
)


SYNTHETIC_PATIENT_PREFIX = "SYN-BRCA-"
TEMPORAL_PATIENT_PREFIX = "SYN-TEMP-BRCA-"
SYNTHETIC_SOURCE = "synthetic_breast_journey_demo"
TEMPORAL_SOURCE = "synthetic_temporal_breast_journey_demo"


def generate_synthetic_breast_cancer_journeys(db, count=25, seed=42):
    rng = random.Random(seed)
    created = {
        "patients_created": 0,
        "patients_skipped": 0,
        "profiles_created": 0,
        "treatments_created": 0,
        "labs_created": 0,
        "symptoms_created": 0,
        "imaging_reports_created": 0,
        "source": SYNTHETIC_SOURCE,
    }

    for index in range(1, count + 1):
        patient_id = f"{SYNTHETIC_PATIENT_PREFIX}{index:04d}"
        if db.query(Patient).filter(Patient.id == patient_id).first():
            created["patients_skipped"] += 1
            continue

        plan = build_synthetic_journey_plan(index=index, rng=rng)
        db.add(Patient(
            id=patient_id,
            name=f"Synthetic Breast Journey {index:04d}",
            diagnosis="Breast cancer - synthetic demo patient",
        ))
        created["patients_created"] += 1

        db.add(BreastCancerProfile(
            patient_id=patient_id,
            cancer_stage=plan["cancer_stage"],
            er_status=plan["er_status"],
            pr_status=plan["pr_status"],
            her2_status=plan["her2_status"],
            molecular_subtype=plan["molecular_subtype"],
            treatment_intent="synthetic neoadjuvant monitoring demo",
            menopausal_status=plan["menopausal_status"],
        ))
        created["profiles_created"] += 1

        baseline_date = date(2025, 1, 1) + timedelta(days=index * 3)
        current_size = plan["baseline_size_cm"]
        _add_imaging_report(
            db=db,
            patient_id=patient_id,
            report_date=baseline_date,
            report_type="Synthetic baseline breast MRI",
            size_cm=current_size,
            response_wording="baseline known malignancy",
            breast_side=plan["breast_side"],
            location=plan["location"],
        )
        created["imaging_reports_created"] += 1

        db.add(LabResult(
            patient_id=patient_id,
            date=baseline_date - timedelta(days=2),
            wbc=round(plan["baseline_wbc"], 1),
            hemoglobin=round(plan["baseline_hgb"], 1),
            platelets=round(plan["baseline_platelets"], 0),
            source=SYNTHETIC_SOURCE,
            source_note="Synthetic longitudinal breast cancer journey data for demo/testing only.",
        ))
        created["labs_created"] += 1

        for cycle in range(1, 5):
            treatment_date = baseline_date + timedelta(days=21 * cycle)
            db.add(Treatment(
                patient_id=patient_id,
                date=treatment_date,
                cycle=cycle,
                drug=plan["regimen"],
            ))
            created["treatments_created"] += 1

            wbc_nadir = max(1.5, plan["baseline_wbc"] - rng.uniform(1.5, 3.8) - (0.15 * cycle))
            hgb_nadir = max(8.2, plan["baseline_hgb"] - rng.uniform(0.4, 1.4) - (0.1 * cycle))
            platelet_nadir = max(75, plan["baseline_platelets"] - rng.uniform(35, 105) - (5 * cycle))
            db.add(LabResult(
                patient_id=patient_id,
                date=treatment_date + timedelta(days=10),
                wbc=round(wbc_nadir, 1),
                hemoglobin=round(hgb_nadir, 1),
                platelets=round(platelet_nadir, 0),
                source=SYNTHETIC_SOURCE,
                source_note="Synthetic post-treatment CBC nadir for demo/testing only.",
            ))
            db.add(LabResult(
                patient_id=patient_id,
                date=treatment_date + timedelta(days=18),
                wbc=round(min(plan["baseline_wbc"], wbc_nadir + rng.uniform(1.1, 2.3)), 1),
                hemoglobin=round(min(plan["baseline_hgb"], hgb_nadir + rng.uniform(0.2, 0.7)), 1),
                platelets=round(min(plan["baseline_platelets"], platelet_nadir + rng.uniform(35, 85)), 0),
                source=SYNTHETIC_SOURCE,
                source_note="Synthetic CBC recovery value for demo/testing only.",
            ))
            created["labs_created"] += 2

            if cycle in plan["symptom_cycles"]:
                db.add(SymptomReport(
                    patient_id=patient_id,
                    date=treatment_date + timedelta(days=rng.randint(3, 12)),
                    symptom=rng.choice(["fatigue", "nausea", "neuropathy", "low appetite", "breast pain"]),
                    severity=rng.randint(2, 8),
                    notes="Synthetic symptom report for workflow testing only.",
                ))
                created["symptoms_created"] += 1

            if cycle in (2, 4):
                current_size = _next_tumor_size(current_size, plan["response"], rng)
                _add_imaging_report(
                    db=db,
                    patient_id=patient_id,
                    report_date=treatment_date + timedelta(days=14),
                    report_type=f"Synthetic follow-up breast MRI cycle {cycle}",
                    size_cm=current_size,
                    response_wording=_response_wording(plan["response"], cycle),
                    breast_side=plan["breast_side"],
                    location=plan["location"],
                )
                created["imaging_reports_created"] += 1

    db.commit()
    return created


def generate_temporal_breast_cancer_journeys(db, count=12, seed=2026, cycles=6):
    rng = random.Random(seed)
    created = {
        "patients_created": 0,
        "patients_skipped": 0,
        "profiles_created": 0,
        "treatments_created": 0,
        "medications_created": 0,
        "labs_created": 0,
        "symptoms_created": 0,
        "imaging_reports_created": 0,
        "cycles_per_patient": cycles,
        "source": TEMPORAL_SOURCE,
    }

    for index in range(1, count + 1):
        patient_id = f"{TEMPORAL_PATIENT_PREFIX}{index:04d}"
        if db.query(Patient).filter(Patient.id == patient_id).first():
            created["patients_skipped"] += 1
            continue

        plan = build_synthetic_journey_plan(index=index + 1000, rng=rng)
        baseline_date = date(2025, 3, 1) + timedelta(days=index * 4)
        current_size = plan["baseline_size_cm"]
        current_wbc = plan["baseline_wbc"]
        current_hgb = plan["baseline_hgb"]
        current_platelets = plan["baseline_platelets"]

        db.add(Patient(
            id=patient_id,
            name=f"Temporal Breast Journey {index:04d}",
            diagnosis="Breast cancer - temporal synthetic monitoring demo",
        ))
        created["patients_created"] += 1

        db.add(BreastCancerProfile(
            patient_id=patient_id,
            cancer_stage=plan["cancer_stage"],
            er_status=plan["er_status"],
            pr_status=plan["pr_status"],
            her2_status=plan["her2_status"],
            molecular_subtype=plan["molecular_subtype"],
            treatment_intent="synthetic neoadjuvant temporal monitoring demo",
            menopausal_status=plan["menopausal_status"],
        ))
        created["profiles_created"] += 1

        _add_temporal_lab(
            db, patient_id, baseline_date - timedelta(days=3),
            current_wbc, current_hgb, current_platelets,
            "Synthetic baseline CBC before treatment start.",
        )
        created["labs_created"] += 1

        _add_imaging_report(
            db=db,
            patient_id=patient_id,
            report_date=baseline_date,
            report_type="Synthetic temporal baseline breast MRI",
            size_cm=current_size,
            response_wording="baseline known malignancy before neoadjuvant treatment",
            breast_side=plan["breast_side"],
            location=plan["location"],
        )
        created["imaging_reports_created"] += 1

        for cycle in range(1, cycles + 1):
            treatment_date = baseline_date + timedelta(days=7 + (cycle - 1) * 21)
            pre_cycle_wbc = max(2.8, min(plan["baseline_wbc"] + 0.6, current_wbc + rng.uniform(-0.2, 0.7)))
            pre_cycle_hgb = max(8.8, min(plan["baseline_hgb"] + 0.2, current_hgb + rng.uniform(-0.1, 0.3)))
            pre_cycle_platelets = max(95, min(plan["baseline_platelets"] + 20, current_platelets + rng.uniform(-8, 35)))

            _add_temporal_lab(
                db, patient_id, treatment_date - timedelta(days=1),
                pre_cycle_wbc, pre_cycle_hgb, pre_cycle_platelets,
                f"Synthetic pre-cycle {cycle} CBC for treatment readiness workflow.",
            )
            created["labs_created"] += 1

            db.add(Treatment(
                patient_id=patient_id,
                date=treatment_date,
                cycle=cycle,
                drug=plan["regimen"],
            ))
            created["treatments_created"] += 1

            for med in _cycle_medications(plan["regimen"], cycle, treatment_date, rng):
                db.add(MedicationLog(patient_id=patient_id, **med))
                created["medications_created"] += 1

            wbc_nadir = max(0.9, pre_cycle_wbc - rng.uniform(1.6, 4.1) - (0.12 * cycle))
            hgb_nadir = max(7.6, pre_cycle_hgb - rng.uniform(0.3, 1.1) - (0.08 * cycle))
            platelet_nadir = max(45, pre_cycle_platelets - rng.uniform(35, 115) - (4 * cycle))
            _add_temporal_lab(
                db, patient_id, treatment_date + timedelta(days=9),
                wbc_nadir, hgb_nadir, platelet_nadir,
                f"Synthetic cycle {cycle} CBC nadir.",
            )
            created["labs_created"] += 1

            recovery_factor = 0.5 if wbc_nadir < 2.0 else 0.75
            current_wbc = min(plan["baseline_wbc"] + 0.4, wbc_nadir + rng.uniform(1.0, 2.6) * recovery_factor)
            current_hgb = min(plan["baseline_hgb"], hgb_nadir + rng.uniform(0.1, 0.6))
            current_platelets = min(plan["baseline_platelets"] + 15, platelet_nadir + rng.uniform(35, 95))
            _add_temporal_lab(
                db, patient_id, treatment_date + timedelta(days=18),
                current_wbc, current_hgb, current_platelets,
                f"Synthetic cycle {cycle} CBC recovery check.",
            )
            created["labs_created"] += 1

            if _should_add_symptom(wbc_nadir, hgb_nadir, cycle, rng):
                symptom, severity = _temporal_symptom(wbc_nadir, hgb_nadir, platelet_nadir, rng)
                db.add(SymptomReport(
                    patient_id=patient_id,
                    date=treatment_date + timedelta(days=rng.randint(3, 12)),
                    symptom=symptom,
                    severity=severity,
                    notes=f"Synthetic symptom during cycle {cycle}; generated for temporal workflow testing.",
                ))
                created["symptoms_created"] += 1

            if cycle in {max(2, cycles // 2), cycles}:
                current_size = _next_tumor_size(current_size, plan["response"], rng)
                _add_imaging_report(
                    db=db,
                    patient_id=patient_id,
                    report_date=treatment_date + timedelta(days=15),
                    report_type=f"Synthetic temporal follow-up breast MRI cycle {cycle}",
                    size_cm=current_size,
                    response_wording=_response_wording(plan["response"], cycle),
                    breast_side=plan["breast_side"],
                    location=plan["location"],
                )
                created["imaging_reports_created"] += 1

        if plan["molecular_subtype"].startswith("HR-positive"):
            db.add(MedicationLog(
                patient_id=patient_id,
                date=baseline_date + timedelta(days=7 + cycles * 21),
                medication=rng.choice(["tamoxifen", "letrozole", "anastrozole"]),
                dose=rng.choice(["20 mg", "2.5 mg", "1 mg"]),
                frequency="daily",
                notes="Synthetic endocrine therapy entry after chemotherapy course.",
                source=TEMPORAL_SOURCE,
            ))
            created["medications_created"] += 1

    db.commit()
    return created


def build_synthetic_journey_plan(index, rng):
    er_status = rng.choice(["Positive", "Positive", "Negative"])
    pr_status = "Positive" if er_status == "Positive" and rng.random() > 0.25 else "Negative"
    her2_status = rng.choice(["Not amplified", "Not amplified", "Amplified"])

    return {
        "cancer_stage": rng.choice(["I", "IIA", "IIB", "IIIA"]),
        "er_status": er_status,
        "pr_status": pr_status,
        "her2_status": her2_status,
        "molecular_subtype": infer_synthetic_subtype(er_status, pr_status, her2_status),
        "menopausal_status": rng.choice(["premenopausal", "postmenopausal", "unknown"]),
        "baseline_size_cm": round(rng.uniform(1.4, 5.8), 1),
        "baseline_wbc": rng.uniform(5.2, 8.4),
        "baseline_hgb": rng.uniform(11.7, 13.8),
        "baseline_platelets": rng.uniform(205, 330),
        "breast_side": rng.choice(["left", "right"]),
        "location": rng.choice(["upper outer quadrant", "upper inner quadrant", "lower outer quadrant", "central breast"]),
        "response": rng.choice(["PR", "PR", "SD", "PCR", "nPCR", "PD"]),
        "regimen": rng.choice(["AC-T", "paclitaxel + carboplatin", "TCHP", "dose-dense AC then paclitaxel"]),
        "symptom_cycles": set(rng.sample([1, 2, 3, 4], k=rng.randint(1, 3))),
    }


def infer_synthetic_subtype(er_status, pr_status, her2_status):
    hormone_positive = er_status == "Positive" or pr_status == "Positive"
    her2_positive = her2_status == "Amplified"
    if hormone_positive and her2_positive:
        return "HR-positive / HER2-positive"
    if hormone_positive:
        return "HR-positive / HER2-negative"
    if her2_positive:
        return "HER2-positive"
    return "triple-negative"


def _next_tumor_size(current_size, response, rng):
    if response == "PCR":
        return max(0.0, round(current_size * rng.uniform(0.1, 0.35), 1))
    if response in {"PR", "nPCR"}:
        return max(0.4, round(current_size * rng.uniform(0.55, 0.82), 1))
    if response == "PD":
        return round(current_size * rng.uniform(1.08, 1.28), 1)
    return round(current_size * rng.uniform(0.9, 1.08), 1)


def _response_wording(response, cycle):
    if response == "PCR" and cycle == 4:
        return "marked interval decrease with no measurable residual enhancing mass"
    if response in {"PR", "nPCR"}:
        return "interval decrease in size and enhancement"
    if response == "PD":
        return "interval increase in size concerning for progression"
    return "overall stable disease"


def _add_imaging_report(db, patient_id, report_date, report_type, size_cm, response_wording, breast_side, location):
    db.add(ImagingReport(
        patient_id=patient_id,
        date=report_date,
        modality="Breast MRI",
        report_type=report_type,
        body_site="Breast",
        findings=(
            f"Synthetic report: {breast_side} breast {location} enhancing mass "
            f"measuring {size_cm} cm. BI-RADS 6. {response_wording}."
        ),
        impression=(
            "Synthetic breast MRI monitoring report generated for software testing only. "
            "Not real patient imaging."
        ),
    ))


def _add_temporal_lab(db, patient_id, lab_date, wbc, hemoglobin, platelets, note):
    db.add(LabResult(
        patient_id=patient_id,
        date=lab_date,
        wbc=round(wbc, 1),
        hemoglobin=round(hemoglobin, 1),
        platelets=round(platelets, 0),
        source=TEMPORAL_SOURCE,
        source_note=note,
    ))


def _cycle_medications(regimen, cycle, treatment_date, rng):
    meds = [
        {
            "date": treatment_date,
            "medication": regimen,
            "dose": "per protocol",
            "frequency": "every 21 days",
            "notes": f"Synthetic treatment administration for cycle {cycle}.",
            "source": TEMPORAL_SOURCE,
        },
        {
            "date": treatment_date,
            "medication": "ondansetron",
            "dose": "8 mg",
            "frequency": "as needed",
            "notes": "Synthetic supportive anti-nausea medication.",
            "source": TEMPORAL_SOURCE,
        },
    ]
    if rng.random() < 0.75:
        meds.append({
            "date": treatment_date + timedelta(days=1),
            "medication": "dexamethasone",
            "dose": "8 mg",
            "frequency": "daily for 2 days",
            "notes": "Synthetic post-infusion supportive medication.",
            "source": TEMPORAL_SOURCE,
        })
    if rng.random() < 0.35:
        meds.append({
            "date": treatment_date + timedelta(days=2),
            "medication": "pegfilgrastim",
            "dose": "6 mg",
            "frequency": "once after cycle",
            "notes": "Synthetic growth-factor support entry.",
            "source": TEMPORAL_SOURCE,
        })
    return meds


def _should_add_symptom(wbc_nadir, hgb_nadir, cycle, rng):
    risk_weight = 0.35
    if wbc_nadir < 2.5:
        risk_weight += 0.25
    if hgb_nadir < 9.5:
        risk_weight += 0.15
    if cycle >= 4:
        risk_weight += 0.10
    return rng.random() < min(0.9, risk_weight)


def _temporal_symptom(wbc_nadir, hgb_nadir, platelet_nadir, rng):
    if wbc_nadir < 1.8:
        return rng.choice(["fever", "fatigue", "chills"]), rng.randint(6, 9)
    if hgb_nadir < 9.2:
        return rng.choice(["fatigue", "shortness of breath", "weakness"]), rng.randint(5, 8)
    if platelet_nadir < 75:
        return rng.choice(["bruising", "bleeding", "fatigue"]), rng.randint(5, 8)
    return rng.choice(["fatigue", "nausea", "neuropathy", "low appetite"]), rng.randint(2, 7)
