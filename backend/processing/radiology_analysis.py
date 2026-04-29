import re


def extract_tumor_size(text):
    match = re.search(r"(\d+\.?\d*)\s*cm", text.lower())
    if match:
        return float(match.group(1))
    return None


def analyze_radiology_reports(ct_df):
    ct_df = ct_df.sort_values("date")

    baseline = ct_df.iloc[0]
    latest = ct_df.iloc[-1]

    baseline_size = extract_tumor_size(baseline["findings"])
    latest_size = extract_tumor_size(latest["findings"])

    if baseline_size and latest_size:
        change = latest_size - baseline_size
        percent_change = (change / baseline_size) * 100

        if change < 0:
            status = "decreased"
        elif change > 0:
            status = "increased"
        else:
            status = "stable"
    else:
        change = None
        percent_change = None
        status = "unknown"

    return {
        "baseline_date": str(baseline["date"]),
        "latest_date": str(latest["date"]),
        "baseline_tumor_size_cm": baseline_size,
        "latest_tumor_size_cm": latest_size,
        "change_cm": change,
        "percent_change": round(percent_change, 1) if percent_change is not None else None,
        "status": status,
        "latest_impression": latest["impression"],
    }
