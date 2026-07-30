"""Calibration diagnostics for the synthetic admin evaluation surface."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedShuffleSplit

from backend.services.admin_metric_interpretation import _ece_status, _round

def _calibration_metrics(labels, probabilities, bins=10):
    if len(labels) == 0:
        return {"status": "unavailable", "bins": []}

    bin_rows = []
    total = len(labels)
    expected_calibration_error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        if index == bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        count = int(mask.sum())
        if count:
            mean_probability = float(probabilities[mask].mean())
            observed_rate = float(labels[mask].mean())
            gap = abs(observed_rate - mean_probability)
            expected_calibration_error += (count / total) * gap
            bin_rows.append({
                "range": f"{lower:.1f}-{upper:.1f}",
                "count": count,
                "mean_probability": _round(mean_probability),
                "observed_positive_rate": _round(observed_rate),
                "gap": _round(gap),
            })

    ece = _round(expected_calibration_error)
    posthoc = _posthoc_calibration_diagnostics(labels, probabilities, bins=bins)
    return {
        "status": _ece_status(expected_calibration_error),
        "purpose": "Checks whether predicted probabilities behave like real probabilities, not just rankings.",
        "expected_calibration_error": ece,
        "brier_score": _round(brier_score_loss(labels, probabilities)),
        "bins": bin_rows,
        "posthoc_calibration": posthoc,
        "recommendation": _calibration_recommendation(expected_calibration_error, posthoc),
    }


def _posthoc_calibration_diagnostics(labels, probabilities, bins=10):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(labels) < 30 or len(np.unique(labels)) < 2:
        return {
            "status": "unavailable",
            "method": "heldout_posthoc_calibration",
            "message": "At least 30 labeled predictions with both classes are needed for a useful calibration split.",
            "candidates": [],
        }

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    try:
        calibration_index, validation_index = next(splitter.split(probabilities.reshape(-1, 1), labels))
    except ValueError as exc:
        return {
            "status": "unavailable",
            "method": "heldout_posthoc_calibration",
            "message": f"Could not create stratified calibration split: {exc}",
            "candidates": [],
        }

    calibration_labels = labels[calibration_index]
    validation_labels = labels[validation_index]
    calibration_probabilities = probabilities[calibration_index]
    validation_probabilities = probabilities[validation_index]

    candidates = [
        _calibration_candidate(
            "raw_validation",
            validation_labels,
            validation_probabilities,
            bins=bins,
            note="Uncalibrated validation probabilities.",
        )
    ]

    try:
        platt = LogisticRegression(solver="lbfgs", random_state=42)
        platt.fit(calibration_probabilities.reshape(-1, 1), calibration_labels)
        platt_probabilities = platt.predict_proba(validation_probabilities.reshape(-1, 1))[:, 1]
        candidates.append(_calibration_candidate(
            "platt_scaling",
            validation_labels,
            platt_probabilities,
            bins=bins,
            note="Logistic calibration fitted on the calibration split.",
        ))
    except ValueError as exc:
        candidates.append({
            "method": "platt_scaling",
            "status": "unavailable",
            "error": str(exc),
        })

    try:
        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(calibration_probabilities, calibration_labels)
        isotonic_probabilities = np.clip(isotonic.transform(validation_probabilities), 0.0, 1.0)
        candidates.append(_calibration_candidate(
            "isotonic_regression",
            validation_labels,
            isotonic_probabilities,
            bins=bins,
            note="Non-parametric calibration fitted on the calibration split.",
        ))
    except ValueError as exc:
        candidates.append({
            "method": "isotonic_regression",
            "status": "unavailable",
            "error": str(exc),
        })

    valid_candidates = [candidate for candidate in candidates if candidate.get("expected_calibration_error") is not None]
    best = min(
        valid_candidates,
        key=lambda candidate: (
            candidate["expected_calibration_error"],
            candidate.get("brier_score") if candidate.get("brier_score") is not None else float("inf"),
        ),
    ) if valid_candidates else None

    return {
        "status": best.get("status") if best else "unavailable",
        "method": "heldout_posthoc_calibration",
        "calibration_patients": int(len(calibration_labels)),
        "validation_patients": int(len(validation_labels)),
        "best_method": best.get("method") if best else None,
        "best_validation_ece": best.get("expected_calibration_error") if best else None,
        "best_validation_brier_score": best.get("brier_score") if best else None,
        "candidates": candidates,
        "claim_boundary": (
            "Post-hoc calibration is a diagnostic and candidate promotion step. "
            "It needs a locked calibration split or external validation before probability-strength claims."
        ),
    }


def _calibration_candidate(method, labels, probabilities, bins=10, note=None):
    ece = _expected_calibration_error(labels, probabilities, bins=bins)
    return {
        "method": method,
        "status": _ece_status(ece),
        "expected_calibration_error": _round(ece),
        "brier_score": _round(brier_score_loss(labels, probabilities)),
        "note": note,
    }


def _expected_calibration_error(labels, probabilities, bins=10):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(labels) == 0:
        return 1.0
    expected_calibration_error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        if index == bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        count = int(mask.sum())
        if count:
            gap = abs(float(labels[mask].mean()) - float(probabilities[mask].mean()))
            expected_calibration_error += (count / len(labels)) * gap
    return float(expected_calibration_error)


def _calibration_recommendation(expected_calibration_error, posthoc=None):
    if expected_calibration_error <= 0.10:
        return "Raw probabilities are acceptable for PoC monitoring language with clear non-clinical caveats."

    best_ece = (posthoc or {}).get("best_validation_ece")
    best_method = (posthoc or {}).get("best_method")
    if best_ece is not None and best_ece <= 0.10 and best_method != "raw_validation":
        return (
            f"Register a calibrated probability head using {best_method} on a locked calibration split, "
            "then re-run threshold and subgroup checks before stronger claims."
        )

    return (
        "Keep probability language conservative, collect more labeled validation journeys, "
        "and inspect calibration before presenting risk scores as reliable probabilities."
    )


