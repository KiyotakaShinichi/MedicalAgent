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
