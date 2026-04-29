def detect_trend(values):
    if values.iloc[-1] < values.iloc[0]:
        return "decreasing"
    elif values.iloc[-1] > values.iloc[0]:
        return "increasing"
    return "stable"


def analyze_labs(labs_df):
    trends = {}

    trends["wbc"] = detect_trend(labs_df["wbc"])
    trends["hemoglobin"] = detect_trend(labs_df["hemoglobin"])
    trends["platelets"] = detect_trend(labs_df["platelets"])

    return trends