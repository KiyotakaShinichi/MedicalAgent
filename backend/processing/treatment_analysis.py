import pandas as pd


def align_labs_with_treatment(labs_df, treatment_df):
    results = []
    window_days = 13

    for _, treatment_row in treatment_df.iterrows():
        cycle_date = treatment_row["date"]

        # Capture post-treatment nadir/recovery while avoiding the next cycle day.
        window = labs_df[
            (labs_df["date"] >= cycle_date)
            & (labs_df["date"] <= cycle_date + pd.Timedelta(days=window_days))
        ]

        if not window.empty:
            min_wbc = float(window["wbc"].min())
            min_hgb = float(window["hemoglobin"].min())
            min_platelets = float(window["platelets"].min())

            results.append({
                "cycle": int(treatment_row["cycle"]),
                "drug": treatment_row["drug"],
                "cycle_date": str(treatment_row["date"]),
                "monitoring_window_days": window_days,
                "min_wbc_post_cycle": min_wbc,
                "min_hemoglobin_post_cycle": min_hgb,
                "min_platelets_post_cycle": min_platelets,
                "lab_dates_in_window": [str(value) for value in window["date"].tolist()],
            })

    return sorted(results, key=lambda row: row["cycle"])
