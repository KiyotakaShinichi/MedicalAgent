"""Classification and regression metrics, and calibration diagnostics.

A leaf module: it computes numbers and imports nothing else from this package.

Expected-calibration-error and the reliability diagnostics live here rather
than in :mod:`calibration` because they *measure* calibration —
``_binary_metrics`` embeds the diagnostics in its own output — whereas
:mod:`calibration` fits and persists a calibrator. Splitting them the other way
round produced an import cycle between the two modules.

Key insertion order in the returned dicts is contractual: it is the column
order of the emitted metrics files and report tables.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def _binary_metrics(labels, probabilities, prefix=""):
    predictions = (np.asarray(probabilities) >= 0.5).astype(int)
    labels = np.asarray(labels).astype(int)
    tn, fp, fn, tp = _confusion_counts(labels, predictions)
    metrics = {
        f"{prefix}accuracy": round(float(accuracy_score(labels, predictions)), 3),
        f"{prefix}balanced_accuracy": round(float(balanced_accuracy_score(labels, predictions)), 3),
        f"{prefix}f1": round(float(f1_score(labels, predictions, zero_division=0)), 3),
        f"{prefix}precision": round(float(precision_score(labels, predictions, zero_division=0)), 3),
        f"{prefix}sensitivity": round(float(recall_score(labels, predictions, zero_division=0)), 3),
        f"{prefix}specificity": round(float(tn / (tn + fp)), 3) if (tn + fp) else None,
        f"{prefix}brier_score": round(float(brier_score_loss(labels, probabilities)), 3),
        f"{prefix}calibration": _probability_calibration_diagnostics(labels, probabilities),
        f"{prefix}confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }
    if len(set(labels.tolist())) > 1:
        metrics[f"{prefix}roc_auc"] = round(float(roc_auc_score(labels, probabilities)), 3)
        metrics[f"{prefix}average_precision"] = round(float(average_precision_score(labels, probabilities)), 3)
    else:
        metrics[f"{prefix}roc_auc"] = None
        metrics[f"{prefix}average_precision"] = None
    return metrics

def _regression_metrics(labels, predictions, prefix=""):
    labels = np.asarray(labels).astype(float)
    predictions = np.asarray(predictions).astype(float)
    rmse = np.sqrt(mean_squared_error(labels, predictions))
    return {
        f"{prefix}mae": round(float(mean_absolute_error(labels, predictions)), 3),
        f"{prefix}rmse": round(float(rmse), 3),
        f"{prefix}r2": round(float(r2_score(labels, predictions)), 3) if len(labels) >= 2 else None,
    }

def _confusion_counts(labels, predictions):
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return int(tn), int(fp), int(fn), int(tp)

def _regression_selection_score(metrics):
    mae = metrics.get("patient_level_mae")
    rmse = metrics.get("patient_level_rmse")
    if mae is None:
        return float("inf")
    if rmse is None:
        return float(mae)
    return round(float(mae) + (0.15 * float(rmse)), 6)

def _expected_calibration_error(labels, probabilities, bins=10):
    labels = np.asarray(labels).astype(int)
    probabilities = np.asarray(probabilities).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= low) & (probabilities < high)
        if high == 1:
            mask = (probabilities >= low) & (probabilities <= high)
        if not np.any(mask):
            continue
        confidence = float(np.mean(probabilities[mask]))
        accuracy = float(np.mean(labels[mask]))
        ece += (float(np.mean(mask)) * abs(accuracy - confidence))
    return round(ece, 4)

def _probability_calibration_diagnostics(labels, probabilities):
    labels = np.asarray(labels).astype(int)
    probabilities = np.clip(np.asarray(probabilities).astype(float), 1e-5, 1 - 1e-5)
    before = {
        "brier_score": round(float(brier_score_loss(labels, probabilities)), 4),
        "ece": _expected_calibration_error(labels, probabilities),
    }
    best_temperature = 1.0
    best_brier = before["brier_score"]
    best_probs = probabilities
    logits = np.log(probabilities / (1 - probabilities))
    for temperature in np.linspace(0.5, 3.0, 26):
        scaled = 1 / (1 + np.exp(-(logits / temperature)))
        brier = float(brier_score_loss(labels, scaled))
        if brier < best_brier:
            best_brier = brier
            best_temperature = float(temperature)
            best_probs = scaled
    return {
        "before_temperature_scaling": before,
        "after_temperature_scaling": {
            "temperature": round(best_temperature, 3),
            "brier_score": round(best_brier, 4),
            "ece": _expected_calibration_error(labels, best_probs),
        },
        "method": "posthoc_temperature_grid_on_evaluation_split",
    }
