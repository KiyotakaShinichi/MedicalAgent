LAB_THRESHOLDS = {
    "wbc": {
        "watch": 4.0,
        "urgent_review": 3.0,
        "unit": "x10^3/uL",
        "label": "WBC",
        "risk_type": "low_wbc",
        "concern": "possible treatment-related white blood cell suppression",
    },
    "platelets": {
        "watch": 150,
        "urgent_review": 100,
        "unit": "x10^3/uL",
        "label": "Platelets",
        "risk_type": "low_platelets",
        "concern": "possible thrombocytopenia",
    },
    "hemoglobin": {
        "watch": 11,
        "urgent_review": 10,
        "unit": "g/dL",
        "label": "Hemoglobin",
        "risk_type": "low_hemoglobin",
        "concern": "possible anemia",
    },
}


def _severity_for_value(value, thresholds):
    if value < thresholds["urgent_review"]:
        return "urgent_review"
    if value < thresholds["watch"]:
        return "watch"
    return None


def _risk_message(metric_config, value, severity):
    action = "Medical review is recommended." if severity == "urgent_review" else "Continue monitoring and review in clinical context."
    return f"{metric_config['label']} reached {value} {metric_config['unit']}, suggesting {metric_config['concern']}. {action}"


def detect_risks(labs_df):
    risks = []

    for metric, config in LAB_THRESHOLDS.items():
        min_index = labs_df[metric].idxmin()
        min_row = labs_df.loc[min_index]
        min_value = float(min_row[metric])
        severity = _severity_for_value(min_value, config)

        if severity:
            risks.append({
                "type": config["risk_type"],
                "category": "lab",
                "severity": severity,
                "message": _risk_message(config, min_value, severity),
                "evidence": {
                    "metric": metric,
                    "date": str(min_row["date"]),
                    "value": min_value,
                    "unit": config["unit"],
                    "watch_threshold": config["watch"],
                    "urgent_review_threshold": config["urgent_review"],
                },
            })

    return risks


def detect_trend_risk(labs_df):
    risks = []

    wbc_start = float(labs_df["wbc"].iloc[0])
    wbc_min = float(labs_df["wbc"].min())
    if wbc_start <= 0:
        return risks

    drop_percent = ((wbc_start - wbc_min) / wbc_start) * 100

    if drop_percent >= 50:
        risks.append({
            "type": "major_wbc_drop",
            "category": "lab_trend",
            "severity": "urgent_review",
            "message": f"WBC dropped by {drop_percent:.1f}% from baseline. This should be reviewed with the treatment timeline.",
            "evidence": {
                "metric": "wbc",
                "baseline_value": wbc_start,
                "lowest_value": wbc_min,
                "percent_drop": round(drop_percent, 1),
                "unit": LAB_THRESHOLDS["wbc"]["unit"],
            },
        })

    return risks


def detect_symptom_risks(symptoms_df):
    risks = []

    if symptoms_df is None or symptoms_df.empty:
        return risks

    high_symptoms = symptoms_df[symptoms_df["severity"] >= 7]
    for _, row in high_symptoms.iterrows():
        risks.append({
            "type": "high_severity_symptom",
            "category": "symptom",
            "severity": "urgent_review",
            "message": f"Patient-reported {row['symptom']} severity was {int(row['severity'])}/10. Clinical review may be needed.",
            "evidence": {
                "date": str(row["date"]),
                "symptom": row["symptom"],
                "severity_score": int(row["severity"]),
                "notes": row.get("notes"),
            },
        })

    return risks


def detect_clinical_rule_risks(labs_df, symptoms_df, treatments_df):
    """Deterministic multi-signal oncology monitoring rules.

    These are conservative clinical-support flags, not diagnosis or treatment orders.
    """

    risks = []
    risks.extend(_critical_cbc_rules(labs_df))
    risks.extend(_fever_after_treatment_rules(symptoms_df, treatments_df))
    risks.extend(_cbc_symptom_combination_rules(labs_df, symptoms_df))
    return risks


def _critical_cbc_rules(labs_df):
    if labs_df is None or labs_df.empty:
        return []

    risks = []
    latest = labs_df.sort_values("date").iloc[-1]
    baseline = labs_df.sort_values("date").iloc[0]

    min_wbc = labs_df.loc[labs_df["wbc"].idxmin()]
    min_platelets = labs_df.loc[labs_df["platelets"].idxmin()]
    min_hemoglobin = labs_df.loc[labs_df["hemoglobin"].idxmin()]

    if float(min_wbc["wbc"]) < 2.0:
        risks.append({
            "type": "critical_wbc_suppression",
            "category": "deterministic_clinical_rule",
            "severity": "urgent_review",
            "message": "WBC fell below 2.0 x10^3/uL. Oncology review is recommended, especially during active chemotherapy.",
            "evidence": {
                "date": str(min_wbc["date"]),
                "metric": "wbc",
                "value": float(min_wbc["wbc"]),
                "threshold": 2.0,
            },
        })

    if float(min_platelets["platelets"]) < 50:
        risks.append({
            "type": "critical_platelet_suppression",
            "category": "deterministic_clinical_rule",
            "severity": "urgent_review",
            "message": "Platelets fell below 50 x10^3/uL. Bleeding-risk review may be needed.",
            "evidence": {
                "date": str(min_platelets["date"]),
                "metric": "platelets",
                "value": float(min_platelets["platelets"]),
                "threshold": 50,
            },
        })

    hemoglobin_drop = float(baseline["hemoglobin"]) - float(latest["hemoglobin"])
    if hemoglobin_drop >= 2.0 or float(min_hemoglobin["hemoglobin"]) < 8.0:
        risks.append({
            "type": "clinically_significant_hemoglobin_drop",
            "category": "deterministic_clinical_rule",
            "severity": "urgent_review" if float(min_hemoglobin["hemoglobin"]) < 8.0 else "watch",
            "message": (
                f"Hemoglobin changed by {hemoglobin_drop:.1f} g/dL from baseline. "
                "Review anemia trend in the treatment-cycle context."
            ),
            "evidence": {
                "baseline_value": float(baseline["hemoglobin"]),
                "latest_value": float(latest["hemoglobin"]),
                "lowest_value": float(min_hemoglobin["hemoglobin"]),
                "latest_date": str(latest["date"]),
            },
        })

    return risks


def _fever_after_treatment_rules(symptoms_df, treatments_df):
    if symptoms_df is None or symptoms_df.empty or treatments_df is None or treatments_df.empty:
        return []

    risks = []
    fever_rows = symptoms_df[symptoms_df["symptom"].astype(str).str.lower().str.contains("fever|chills", regex=True, na=False)]
    if fever_rows.empty:
        return risks

    treatment_dates = list(treatments_df["date"])
    for _, fever in fever_rows.iterrows():
        fever_date = fever["date"]
        recent_cycles = [
            treatment_date for treatment_date in treatment_dates
            if 0 <= (fever_date - treatment_date).days <= 14
        ]
        if recent_cycles:
            risks.append({
                "type": "fever_after_recent_chemotherapy",
                "category": "deterministic_clinical_rule",
                "severity": "urgent_review",
                "message": "Fever/chills were reported within 14 days after treatment. This should be reviewed urgently in chemotherapy context.",
                "evidence": {
                    "symptom_date": str(fever_date),
                    "severity_score": int(fever.get("severity", 0)),
                    "recent_treatment_dates": [str(value) for value in recent_cycles],
                },
            })

    return risks


def _cbc_symptom_combination_rules(labs_df, symptoms_df):
    if labs_df is None or labs_df.empty or symptoms_df is None or symptoms_df.empty:
        return []

    risks = []
    low_wbc_rows = labs_df[labs_df["wbc"] < 3.0]
    fever_rows = symptoms_df[symptoms_df["symptom"].astype(str).str.lower().str.contains("fever|chills", regex=True, na=False)]
    if low_wbc_rows.empty or fever_rows.empty:
        return risks

    for _, fever in fever_rows.iterrows():
        fever_date = fever["date"]
        nearby_low_wbc = low_wbc_rows[
            low_wbc_rows["date"].apply(lambda lab_date: abs((lab_date - fever_date).days) <= 3)
        ]
        if not nearby_low_wbc.empty:
            lowest = nearby_low_wbc.loc[nearby_low_wbc["wbc"].idxmin()]
            risks.append({
                "type": "fever_with_low_wbc",
                "category": "deterministic_clinical_rule",
                "severity": "urgent_review",
                "message": "Fever/chills occurred near a low WBC result. Clinician review should be prioritized.",
                "evidence": {
                    "symptom_date": str(fever_date),
                    "wbc_date": str(lowest["date"]),
                    "wbc": float(lowest["wbc"]),
                },
            })
            break

    return risks
