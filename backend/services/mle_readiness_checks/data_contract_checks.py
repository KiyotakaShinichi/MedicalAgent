import pandas as pd

from backend.services.mle_readiness_checks.core import check, higher_status, lower_status, rounded


REQUIRED_TEMPORAL_COLUMNS = {
    "patient_id",
    "cycle",
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
    "mri_tumor_size_cm",
    "mri_percent_change_from_baseline",
    "max_symptom_severity",
    "symptom_count",
    "intervention_count",
    "dose_delayed",
    "dose_reduced",
    "treatment_success_binary",
}


NUMERIC_RANGES = {
    "age": (18, 100),
    "cycle": (1, 24),
    "pre_wbc": (0.1, 50),
    "pre_anc": (0.0, 40),
    "pre_hemoglobin": (3, 22),
    "pre_platelets": (5, 1200),
    "nadir_wbc": (0.0, 50),
    "nadir_anc": (0.0, 40),
    "nadir_hemoglobin": (3, 22),
    "nadir_platelets": (5, 1200),
    "mri_tumor_size_cm": (0.0, 20),
    "mri_percent_change_from_baseline": (-100, 200),
    "max_symptom_severity": (0, 10),
    "symptom_count": (0, 50),
    "intervention_count": (0, 50),
    "dose_delayed": (0, 1),
    "dose_reduced": (0, 1),
    "treatment_success_binary": (0, 1),
}


def data_contract_checks(frame):
    if frame is None or frame.empty:
        return [
            check(
                name="training_data_loadable",
                category="data_contract",
                status="failed",
                value="missing_or_empty",
                threshold="non-empty CSV",
                meaning="Training data must load before any MLE gate can be trusted.",
                hard_gate=True,
                remediation="Regenerate or restore the temporal ML training CSV.",
            )
        ]

    checks = []
    missing_columns = sorted(REQUIRED_TEMPORAL_COLUMNS - set(frame.columns))
    checks.append(
        check(
            name="required_columns_present",
            category="data_contract",
            status="passed" if not missing_columns else "failed",
            value={"missing": missing_columns},
            threshold=f"{len(REQUIRED_TEMPORAL_COLUMNS)} required columns",
            meaning="A fixed feature contract protects training and inference code from silent schema drift.",
            hard_gate=True,
            remediation="Update the data generator/exporter or model feature mapping.",
        )
    )

    patient_count = int(frame["patient_id"].nunique()) if "patient_id" in frame.columns else 0
    row_count = int(len(frame))
    checks.append(
        check(
            name="minimum_training_size",
            category="data_contract",
            status=higher_status(patient_count, [100, 200, 300]),
            value={"rows": row_count, "patients": patient_count},
            threshold=">=200 patients preferred, >=100 minimum",
            meaning="A tiny patient split makes metrics unstable.",
            hard_gate=patient_count < 100,
            remediation="Generate more complete synthetic journeys before training.",
        )
    )

    if {"patient_id", "cycle"}.issubset(frame.columns):
        duplicate_count = int(frame.duplicated(subset=["patient_id", "cycle"]).sum())
        cycles_per_patient = frame.groupby("patient_id")["cycle"].nunique()
        depth_rate = float((cycles_per_patient >= 6).mean())
    else:
        duplicate_count = None
        depth_rate = 0.0
    checks.append(
        check(
            name="patient_cycle_uniqueness",
            category="data_contract",
            status="passed" if duplicate_count == 0 else "failed",
            value=duplicate_count,
            threshold="0 duplicate patient-cycle rows",
            meaning="Longitudinal rows should have one feature row per patient/cycle.",
            hard_gate=duplicate_count not in {0, None},
            remediation="Deduplicate temporal rows before training.",
        )
    )
    checks.append(
        check(
            name="longitudinal_depth",
            category="data_contract",
            status=higher_status(depth_rate, [0.80, 0.90, 0.98]),
            value=round(depth_rate, 3),
            threshold=">=0.90 patients with at least 6 cycles",
            meaning="Treatment monitoring needs temporal depth, not just one-row tabular data.",
            hard_gate=depth_rate < 0.80,
            remediation="Regenerate journeys with enough treatment cycles per patient.",
        )
    )

    if "treatment_success_binary" in frame.columns:
        prevalence = float(frame["treatment_success_binary"].dropna().mean())
        status = (
            "passed"
            if 0.35 <= prevalence <= 0.65
            else "acceptable"
            if 0.25 <= prevalence <= 0.75
            else "failed"
        )
    else:
        prevalence = None
        status = "failed"
    checks.append(
        check(
            name="label_prevalence_balance",
            category="data_contract",
            status=status,
            value=round(prevalence, 3) if prevalence is not None else None,
            threshold="preferred 0.35-0.65, minimum 0.25-0.75",
            meaning="Extreme imbalance can make accuracy misleading and destabilize threshold tuning.",
            hard_gate=status == "failed",
            remediation="Rebalance synthetic outcomes or use metrics/thresholds designed for imbalance.",
        )
    )

    missing_rate = float(
        frame[list(REQUIRED_TEMPORAL_COLUMNS & set(frame.columns))].isna().mean().mean()
    )
    checks.append(
        check(
            name="feature_missingness",
            category="data_contract",
            status=lower_status(missing_rate, [0.20, 0.10, 0.05]),
            value=round(missing_rate, 3),
            threshold="<=0.10 acceptable, <=0.05 strong",
            meaning="Missing core longitudinal features weaken training and monitoring confidence.",
            hard_gate=missing_rate > 0.20,
            remediation="Improve data generation/imputation or add missing-data indicators.",
        )
    )

    violations = range_violations(frame)
    checks.append(
        check(
            name="numeric_range_contract",
            category="data_contract",
            status="passed" if not violations else "failed",
            value={"violation_count": len(violations), "examples": violations[:8]},
            threshold="0 out-of-range core numeric values",
            meaning="Range checks catch broken generators, bad units, or corrupt uploads.",
            hard_gate=bool(violations),
            remediation="Fix invalid feature units or tighten input validation.",
        )
    )
    return checks


def range_violations(frame):
    violations = []
    for column, (minimum, maximum) in NUMERIC_RANGES.items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        bad = values[(values < minimum) | (values > maximum)]
        if not bad.empty:
            violations.append(
                {
                    "column": column,
                    "count": int(len(bad)),
                    "min_allowed": minimum,
                    "max_allowed": maximum,
                    "observed_min": rounded(values.min()),
                    "observed_max": rounded(values.max()),
                }
            )
    return violations
