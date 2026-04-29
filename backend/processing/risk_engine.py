def detect_risks(labs_df):
    risks = []
    latest = labs_df.iloc[-1]
    min_wbc = labs_df["wbc"].min()
    min_platelets = labs_df["platelets"].min()
    min_hgb = labs_df["hemoglobin"].min()

    if min_wbc < 3.0:
        risks.append({
            "type": "low_wbc",
            "severity": "warning",
            "message": f"WBC dropped to {min_wbc}, possible neutropenia risk. Medical review recommended."
        })

    if min_platelets < 150:
        risks.append({
            "type": "low_platelets",
            "severity": "warning",
            "message": f"Platelets dropped to {min_platelets}, possible thrombocytopenia risk."
        })

    if min_hgb < 10:
        risks.append({
            "type": "low_hemoglobin",
            "severity": "warning",
            "message": f"Hemoglobin dropped to {min_hgb}, possible anemia risk."
        })

    return risks


def detect_trend_risk(labs_df):
    risks = []

    wbc_start = labs_df["wbc"].iloc[0]
    wbc_min = labs_df["wbc"].min()

    drop_percent = ((wbc_start - wbc_min) / wbc_start) * 100

    if drop_percent >= 50:
        risks.append({
            "type": "major_wbc_drop",
            "severity": "warning",
            "message": f"WBC dropped by {drop_percent:.1f}% from baseline."
        })

    return risks