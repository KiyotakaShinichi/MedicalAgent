"""Row-level model output aggregated into patient-level prediction records.

Models score individual treatment rows; the product reports per patient. These
helpers own that reduction and the record shape written to the prediction
CSVs, which the evaluation and XAI modules parse.
"""



def _base_patient_prediction_rows(test_rows, target):
    return (
        test_rows.groupby("patient_id", as_index=False)[target]
        .max()
        .rename(columns={target: "actual_label"})
        .sort_values("patient_id")
        .reset_index(drop=True)
    )

def _aggregate_patient_predictions(test_rows, target, probabilities, model_name):
    rows = test_rows[["patient_id", target]].copy()
    rows["probability"] = probabilities
    grouped = (
        rows.groupby("patient_id")
        .agg(actual_label=(target, "max"), probability=("probability", "mean"))
        .reset_index()
        .rename(columns={"probability": f"{model_name}_probability"})
    )
    grouped[f"{model_name}_probability"] = grouped[f"{model_name}_probability"].round(6)
    grouped[f"{model_name}_predicted_label"] = (grouped[f"{model_name}_probability"] >= 0.5).astype(int)
    return grouped

def _base_patient_regression_rows(test_rows, target):
    last_rows = (
        test_rows.sort_values(["patient_id", "cycle"])
        .groupby("patient_id", as_index=False)
        .tail(1)
    )
    return (
        last_rows[["patient_id", target]]
        .rename(columns={target: "actual_response_score_percent"})
        .sort_values("patient_id")
        .reset_index(drop=True)
    )

def _aggregate_patient_regression_predictions(test_rows, target, predictions, model_name):
    rows = test_rows[["patient_id", "cycle", target]].copy()
    rows["prediction"] = predictions
    last_rows = (
        rows.sort_values(["patient_id", "cycle"])
        .groupby("patient_id", as_index=False)
        .tail(1)
        .rename(columns={target: "actual_response_score_percent", "prediction": f"{model_name}_response_score_percent"})
    )
    last_rows[f"{model_name}_response_score_percent"] = last_rows[f"{model_name}_response_score_percent"].round(3)
    return (
        last_rows[["patient_id", "actual_response_score_percent", f"{model_name}_response_score_percent"]]
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
