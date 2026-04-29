def detect_risks(labs_df):
    risks = []

    latest = labs_df.iloc[-1]

    if latest["wbc"] < 3.0:
        risks.append("Neutropenia risk")

    if latest["platelets"] < 150:
        risks.append("Low platelet risk")

    return risks

def detect_trend_risk(labs_df):
    risks = []

    # check continuous drop
    if labs_df["wbc"].is_monotonic_decreasing:
        risks.append("Continuous WBC decline")

    return risks