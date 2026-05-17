"""Head-to-head: legacy regressor vs. modality-robust regressor.

Mirror of ``modality_robustness_comparison`` (which did the classification
head) for the response-score regression head.  Runs both regressors against
the same eight modality-dropout scenarios and reports per-scenario MAE
deltas.  Two views per scenario:

  1. **force_score** — both regressors forced to predict on every row
     (abstention rules disabled).  Isolates the intrinsic robustness of
     the trained model.
  2. **with_abstention** — both regressors go through the production
     abstention layer.  When rules abstain, no prediction is made; the
     comparison happens on covered rows only.

Quantile-band coverage is the same for both regressors (it comes from the
shared quantile heads), so the meaningful comparison here is the point-
estimate MAE.

Artifact: ``Data/evals/models/latest_regression_robustness_comparison.json``
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.services.complete_synthetic_training import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)
from backend.services.evidence_abstention_eval import SCENARIOS, _strip_modalities
from backend.services.evidence_sufficiency import assess_evidence
from backend.services.modality_dropout_regression_training import (
    DEFAULT_MODEL_PATH as DEFAULT_ROBUST_REGRESSION_PATH,
)
from backend.services.hybrid_prediction import DEFAULT_REGRESSION_MODEL_PATH


DEFAULT_TRAINING_ROWS_PATH = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_regression_robustness_comparison.json"
DEFAULT_REGRESSION_TARGET = "response_score_percent"

REGRESSORS_UNDER_TEST: dict[str, str] = {
    "legacy": DEFAULT_REGRESSION_MODEL_PATH,
    "robust": DEFAULT_ROBUST_REGRESSION_PATH,
}


def run_regression_robustness_comparison(
    *,
    training_rows_path: str = DEFAULT_TRAINING_ROWS_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    target: str = DEFAULT_REGRESSION_TARGET,
    sample_size: int | None = None,
) -> dict[str, Any]:
    """Run the sweep + write the artifact."""
    rows = pd.read_csv(training_rows_path)
    if target not in rows.columns:
        raise ValueError(f"Target column '{target}' missing from {training_rows_path}")
    rows = rows.dropna(subset=[target]).copy()
    if sample_size is not None and sample_size < len(rows):
        rows = rows.sample(n=sample_size, random_state=0).reset_index(drop=True)

    for label, path in REGRESSORS_UNDER_TEST.items():
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{label} regressor artifact missing at {path}. Run training first.",
            )
    loaded = {label: joblib.load(path) for label, path in REGRESSORS_UNDER_TEST.items()}

    scenario_reports: list[dict[str, Any]] = []
    for scenario, stripped in SCENARIOS.items():
        masked = _strip_modalities(rows, stripped)
        scenario_reports.append(
            _compare_one_scenario(
                scenario=scenario,
                stripped=stripped,
                masked_rows=masked,
                target=target,
                models=loaded,
            ),
        )

    payload = _build_payload(scenario_reports, training_rows_path, output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _compare_one_scenario(
    *,
    scenario: str,
    stripped: tuple[str, ...],
    masked_rows: pd.DataFrame,
    target: str,
    models: dict[str, Any],
) -> dict[str, Any]:
    """Two views per scenario for each regressor."""
    y_true = masked_rows[target].astype(float).to_numpy()
    feature_frame = masked_rows[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    force_score_view: dict[str, Any] = {}
    for name, model in models.items():
        preds = model.predict(feature_frame)
        force_score_view[name] = {
            "mae": _round(float(np.mean(np.abs(preds - y_true)))),
            "rmse": _round(float(np.sqrt(np.mean((preds - y_true) ** 2)))),
            "mean_prediction": _round(float(np.mean(preds))),
        }

    # With abstention: filter to covered rows (where the response_regression
    # sufficiency rule would not abstain), then compute MAE only on those.
    covered_view: dict[str, Any] = {}
    covered_indices = [
        idx for idx, row in enumerate(masked_rows.to_dict("records"))
        if not assess_evidence(row, question="response_regression").abstain
    ]
    if covered_indices:
        idx_array = np.array(covered_indices)
        covered_y = y_true[idx_array]
        covered_features = feature_frame.iloc[idx_array]
        for name, model in models.items():
            preds = model.predict(covered_features)
            covered_view[name] = {
                "coverage_rate": _round(len(covered_indices) / max(1, len(masked_rows))),
                "abstention_rate": _round(1.0 - len(covered_indices) / max(1, len(masked_rows))),
                "mae_on_covered": _round(float(np.mean(np.abs(preds - covered_y)))),
                "covered_count": int(len(covered_indices)),
            }
    else:
        # Every row abstains — nothing to compare.  Record the abstention
        # rate so the dashboard shows it cleanly.
        for name in models:
            covered_view[name] = {
                "coverage_rate": 0.0,
                "abstention_rate": 1.0,
                "mae_on_covered": None,
                "covered_count": 0,
            }

    return {
        "scenario": scenario,
        "stripped_modalities": list(stripped),
        "rows_evaluated": int(len(masked_rows)),
        "force_score": force_score_view,
        "with_abstention": covered_view,
        "deltas": {
            # Negative delta = robust is BETTER (lower MAE).
            "force_score_mae_robust_minus_legacy": _round(
                force_score_view["robust"]["mae"] - force_score_view["legacy"]["mae"],
            ),
            "with_abstention_mae_robust_minus_legacy": _covered_delta(covered_view),
        },
    }


def _build_payload(
    scenario_reports: list[dict[str, Any]],
    training_rows_path: str,
    output_path: str,
) -> dict[str, Any]:
    overall = _summarise(scenario_reports)
    return {
        "schema_version": "regression_robustness_comparison_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": overall["status"],
        "training_rows_path": training_rows_path,
        "output_path": output_path,
        "regressors_under_test": REGRESSORS_UNDER_TEST,
        "scenarios": scenario_reports,
        "summary": overall,
        "interpretation": (
            "Negative force_score_mae deltas mean the modality-robust "
            "regressor produces lower error on rows the legacy regressor "
            "struggles with — the analog of the +8.3pp accuracy result "
            "the classifier got from dropout retraining."
        ),
        "claim_boundary": (
            "Synthetic comparison only. Improvements describe how the "
            "regressor handles its own training-distribution missingness "
            "pattern, not clinical regression validity."
        ),
    }


def _summarise(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    full = next((s for s in scenarios if s["scenario"] == "full_data"), None)
    # Robust wins when the MAE delta is negative (lower error).
    wins = sum(
        1 for s in scenarios
        if (s["deltas"]["force_score_mae_robust_minus_legacy"] or 0) < -0.5
    )
    ties = sum(
        1 for s in scenarios
        if abs(s["deltas"]["force_score_mae_robust_minus_legacy"] or 0) <= 0.5
    )
    losses = len(scenarios) - wins - ties

    full_delta = (
        full["deltas"]["force_score_mae_robust_minus_legacy"]
        if full else None
    )
    full_regression = full_delta is not None and full_delta > 2.0
    if full_regression:
        status = "needs_attention"
    elif wins >= losses:
        status = "robust"
    else:
        status = "acceptable"

    return {
        "status": status,
        "scenario_count": len(scenarios),
        "force_score_mae_wins_for_robust": wins,
        "force_score_mae_ties": ties,
        "force_score_mae_losses_for_robust": losses,
        "full_data_mae_delta": full_delta,
    }


def _round(value: float | None, ndigits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)


def _covered_delta(covered_view: dict[str, Any]) -> float | None:
    robust = covered_view["robust"]["mae_on_covered"]
    legacy = covered_view["legacy"]["mae_on_covered"]
    if robust is None or legacy is None:
        return None
    return _round(float(robust) - float(legacy))


def load_regression_robustness_comparison(
    path: str = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {
            "schema_version": "regression_robustness_comparison_v1",
            "status": "missing",
            "message": (
                "Regression-robustness comparison has not been generated yet. "
                "Run the modality-dropout regression trainer + this comparison."
            ),
            "scenarios": [],
            "summary": {},
        }
    return json.loads(file_path.read_text(encoding="utf-8"))


__all__ = [
    "REGRESSORS_UNDER_TEST",
    "load_regression_robustness_comparison",
    "run_regression_robustness_comparison",
]
