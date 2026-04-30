import pandas as pd

def align_labs_with_treatment(labs_df, treatment_df):
    results = []

    for _, treatment_row in treatment_df.iterrows():
        cycle_date = treatment_row["date"]

        # get labs AFTER this cycle (next 7–14 days window)
        window = labs_df[
            (labs_df["date"] >= cycle_date) &
            (labs_df["date"] <= cycle_date + pd.Timedelta(days=14))
        ]

        if not window.empty:
            min_wbc = float(window["wbc"].min())
            min_hgb = float(window["hemoglobin"].min())
            min_platelets = float(window["platelets"].min())

            results.append({
                "cycle": treatment_row["cycle"],
                "drug": treatment_row["drug"],
                "cycle_date": str(treatment_row["date"]),
                "monitoring_window_days": 14,
                "min_wbc_post_cycle": min_wbc,
                "min_hemoglobin_post_cycle": min_hgb,
                "min_platelets_post_cycle": min_platelets,
                "lab_dates_in_window": [str(value) for value in window["date"].tolist()],
            })

    return results
