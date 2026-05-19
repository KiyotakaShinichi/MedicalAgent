from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.services.oncology_canonical_schema import ROOT_DIR


DEFAULT_SYNTHETIC_CSV = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
DEFAULT_BREASTDCEDL_CSV = "Data/breastdcedl_spy1_features.csv"
DEFAULT_CBIOPORTAL_CSV = "Data/external_bridge/cbioportal/canonical_cbioportal_breast_rows.csv"
DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_common_feature_transfer_stress.json"

COMMON_FEATURES = ["age", "baseline_tumor_size_mm", "hr_positive", "her2_positive", "triple_negative"]

CLAIM_BOUNDARY = (
    "Common-feature transfer stress is an engineering domain-shift check over fields shared by synthetic "
    "OncoTrack rows and public breast-cancer cohorts. Synthetic treatment-success, BreastDCEDL pCR, and "
    "cBioPortal survival/recurrence labels are not the same endpoint. This artifact must not be described as "
    "clinical validation, treatment efficacy evidence, or real patient prediction performance."
)


def run_common_feature_transfer_stress(
    *,
    synthetic_csv: str = DEFAULT_SYNTHETIC_CSV,
    breastdcedl_csv: str = DEFAULT_BREASTDCEDL_CSV,
    cbioportal_csv: str = DEFAULT_CBIOPORTAL_CSV,
    output_path: str = DEFAULT_OUTPUT_PATH,
    seed: int = 20260519,
) -> dict[str, Any]:
    synthetic = _synthetic_patient_level(_read_frame(synthetic_csv))
    breastdcedl = _breastdcedl_rows(_read_frame(breastdcedl_csv))
    cbioportal = _cbioportal_rows(_read_frame(cbioportal_csv))

    synthetic_fit = _fit_and_evaluate(synthetic, label_col="label", seed=seed)
    breastdcedl_fit = _fit_and_evaluate(breastdcedl, label_col="label", seed=seed)

    transfer_results: dict[str, Any] = {}
    if synthetic_fit["model"] is not None:
        transfer_results["synthetic_model_on_breastdcedl"] = _score_transfer(
            synthetic_fit["model"],
            breastdcedl,
            label_col="label",
            label_context="BreastDCEDL pCR label; not equivalent to OncoTrack treatment_success_binary.",
        )
        transfer_results["synthetic_model_on_cbioportal"] = _score_transfer(
            synthetic_fit["model"],
            cbioportal,
            label_col=None,
            label_context="cBioPortal rows are scored for probability-distribution stress only.",
        )
    if breastdcedl_fit["model"] is not None:
        transfer_results["breastdcedl_pcr_model_on_synthetic"] = _score_transfer(
            breastdcedl_fit["model"],
            synthetic,
            label_col="label",
            label_context="Synthetic treatment_success_binary is not equivalent to pCR.",
        )

    domain_shift = {
        "synthetic_vs_breastdcedl": _domain_shift(synthetic, breastdcedl),
        "synthetic_vs_cbioportal": _domain_shift(synthetic, cbioportal),
    }
    warnings = _warnings(domain_shift, transfer_results)
    status = "strong" if synthetic_fit["metrics"]["status"] == "computed" and breastdcedl_fit["metrics"]["status"] == "computed" else "needs_attention"

    payload = {
        "schema_version": "common_feature_transfer_stress_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "feature_set": COMMON_FEATURES,
        "cohort_sizes": {
            "synthetic_patient_level": int(len(synthetic)),
            "breastdcedl_spy1": int(len(breastdcedl)),
            "cbioportal_tcga_metabric": int(len(cbioportal)),
        },
        "within_dataset_models": {
            "synthetic_treatment_success": synthetic_fit["metrics"],
            "breastdcedl_pcr": breastdcedl_fit["metrics"],
        },
        "transfer_stress": transfer_results,
        "domain_shift": domain_shift,
        "warnings": warnings,
        "promotion_decision": {
            "recommendation": "hold_monitor_only",
            "promotion_allowed": False,
            "reason": (
                "The shared feature set can expose distribution mismatch and brittle transfer behavior, "
                "but the source and target endpoints are not clinically interchangeable."
            ),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(_resolve(output_path), payload)
    return payload


def _synthetic_patient_level(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_common_frame()
    final = frame.sort_values(["patient_id", "cycle"]).groupby("patient_id", as_index=False).tail(1).copy()
    subtype = final.get("molecular_subtype", pd.Series(index=final.index, dtype=str)).astype(str)
    final["baseline_tumor_size_mm"] = pd.to_numeric(final.get("mri_tumor_size_cm"), errors="coerce") * 10.0
    final["hr_positive"] = subtype.str.contains("HR\\+", case=False, regex=True).astype(int)
    final["her2_positive"] = subtype.str.contains("HER2\\+", case=False, regex=True).astype(int)
    final["triple_negative"] = subtype.str.contains("triple", case=False, regex=False).astype(int)
    final["label"] = pd.to_numeric(final.get("treatment_success_binary"), errors="coerce")
    return final[COMMON_FEATURES + ["label"]].dropna(subset=["label"])


def _breastdcedl_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_common_frame()
    out = frame.copy()
    out["age"] = pd.to_numeric(out.get("age"), errors="coerce")
    if "baseline_longest_diameter_mm" in out.columns:
        out["baseline_tumor_size_mm"] = pd.to_numeric(out["baseline_longest_diameter_mm"], errors="coerce")
    elif "imaging_features" in out.columns:
        out["baseline_tumor_size_mm"] = out["imaging_features"].map(_extract_baseline_size)
    else:
        out["baseline_tumor_size_mm"] = np.nan
    subtype = out.get("molecular_subtype", pd.Series(index=out.index, dtype=str)).astype(str).str.lower()
    er_status = _string_series(out, "er_status").str.lower()
    pr_status = _string_series(out, "pr_status").str.lower()
    her2_status = _string_series(out, "her2_status").str.lower()
    out["hr_positive"] = (
        subtype.str.contains("hrpos", regex=False)
        | subtype.str.contains("lum", regex=False)
        | (er_status == "positive")
        | (pr_status == "positive")
    ).astype(int)
    out["her2_positive"] = (
        subtype.str.contains("her2pos", regex=False)
        | subtype.str.contains("her2\\+", regex=True)
        | (her2_status == "positive")
    ).astype(int)
    out["triple_negative"] = (
        subtype.str.contains("tripleneg", regex=False)
        | subtype.str.contains("triple", regex=False)
    ).astype(int)
    if "pcr_label" in out.columns:
        out["label"] = pd.to_numeric(out["pcr_label"], errors="coerce")
    elif "outcome_label_name" in out.columns and "outcome_label_value" in out.columns:
        is_pcr = out["outcome_label_name"].astype(str).str.lower() == "pcr"
        out["label"] = pd.to_numeric(out["outcome_label_value"].where(is_pcr), errors="coerce")
    else:
        out["label"] = np.nan
    return out[COMMON_FEATURES + ["label"]].dropna(subset=["label"])


def _cbioportal_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_common_frame(include_label=False)
    out = frame.copy()
    out["age"] = pd.to_numeric(out.get("age"), errors="coerce")
    out["baseline_tumor_size_mm"] = out.get("imaging_features", pd.Series(index=out.index, dtype=object)).map(_extract_baseline_size)
    er = out.get("er_status", pd.Series(index=out.index, dtype=str)).astype(str).str.lower()
    pr = out.get("pr_status", pd.Series(index=out.index, dtype=str)).astype(str).str.lower()
    her2 = out.get("her2_status", pd.Series(index=out.index, dtype=str)).astype(str).str.lower()
    subtype = out.get("molecular_subtype", pd.Series(index=out.index, dtype=str)).astype(str).str.lower()
    out["hr_positive"] = ((er == "positive") | (pr == "positive") | subtype.str.contains("luma|lumb|luminal", regex=True)).astype(int)
    out["her2_positive"] = ((her2 == "positive") | subtype.str.contains("her2", regex=False)).astype(int)
    out["triple_negative"] = (
        subtype.str.contains("basal|triple", regex=True)
        | ((er == "negative") & (pr == "negative") & (her2 == "negative"))
    ).astype(int)
    return out[COMMON_FEATURES]


def _fit_and_evaluate(frame: pd.DataFrame, *, label_col: str, seed: int) -> dict[str, Any]:
    y = pd.to_numeric(frame.get(label_col), errors="coerce").dropna().astype(int)
    frame = frame.loc[y.index].copy()
    if len(frame) < 20 or y.nunique() < 2:
        return {
            "model": None,
            "metrics": {"status": "not_computed", "reason": "too few rows or only one label class", "rows": int(len(frame))},
        }
    train, test = train_test_split(frame, test_size=0.30, random_state=seed, stratify=y)
    model = _model()
    model.fit(train[COMMON_FEATURES], train[label_col].astype(int))
    probs = model.predict_proba(test[COMMON_FEATURES])[:, 1]
    labels = test[label_col].astype(int).to_numpy()
    preds = (probs >= 0.5).astype(int)
    return {
        "model": model,
        "metrics": {
            "status": "computed",
            "rows": int(len(frame)),
            "test_rows": int(len(test)),
            "positive_rate": round(float(y.mean()), 4),
            "roc_auc": _safe_auc(labels, probs),
            "brier": round(float(brier_score_loss(labels, probs)), 4),
            "accuracy": round(float(accuracy_score(labels, preds)), 4),
        },
    }


def _score_transfer(
    model: Pipeline,
    frame: pd.DataFrame,
    *,
    label_col: str | None,
    label_context: str,
) -> dict[str, Any]:
    if frame.empty:
        return {"status": "not_computed", "reason": "empty target cohort", "label_context": label_context}
    probs = model.predict_proba(frame[COMMON_FEATURES])[:, 1]
    result: dict[str, Any] = {
        "status": "computed",
        "rows": int(len(frame)),
        "probability_distribution": _probability_summary(probs),
        "label_context": label_context,
        "clinical_interpretation_allowed": False,
    }
    if label_col and label_col in frame.columns and frame[label_col].notna().any():
        labels = pd.to_numeric(frame[label_col], errors="coerce")
        mask = labels.notna()
        if mask.sum() >= 20 and labels[mask].nunique() >= 2:
            result["mismatched_endpoint_metrics"] = {
                "roc_auc_against_non_equivalent_label": _safe_auc(labels[mask].astype(int).to_numpy(), probs[mask.to_numpy()]),
                "brier_against_non_equivalent_label": round(float(brier_score_loss(labels[mask].astype(int), probs[mask.to_numpy()])), 4),
                "warning": "Computed only to expose transfer stress; not a valid clinical validation metric.",
            }
    return result


def _domain_shift(source: pd.DataFrame, target: pd.DataFrame) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for feature in COMMON_FEATURES:
        s = pd.to_numeric(source.get(feature), errors="coerce").dropna()
        t = pd.to_numeric(target.get(feature), errors="coerce").dropna()
        rows[feature] = {
            "source_available": int(len(s)),
            "target_available": int(len(t)),
            "source_mean": _rounded_mean(s),
            "target_mean": _rounded_mean(t),
            "absolute_mean_delta": round(abs(float(s.mean() - t.mean())), 4) if len(s) and len(t) else None,
            "psi": _psi(s, t),
            "source_missing_rate": round(float(source[feature].isna().mean()), 4) if feature in source else 1.0,
            "target_missing_rate": round(float(target[feature].isna().mean()), 4) if feature in target else 1.0,
        }
    return rows


def _warnings(domain_shift: dict[str, Any], transfer_results: dict[str, Any]) -> list[str]:
    warnings = [
        "Synthetic treatment success and public pCR/survival-style endpoints are not interchangeable.",
        "Transfer probabilities are domain-stress probes only and must not be patient-facing.",
    ]
    for comparison, rows in domain_shift.items():
        for feature, metrics in rows.items():
            if metrics.get("psi") is not None and metrics["psi"] >= 0.25:
                warnings.append(f"{comparison}:{feature} has large PSI-style distribution shift ({metrics['psi']}).")
            if metrics.get("target_missing_rate", 0) >= 0.8:
                warnings.append(f"{comparison}:{feature} is mostly missing in the target cohort.")
    for name, result in transfer_results.items():
        distribution = result.get("probability_distribution", {})
        if distribution.get("p10") is not None and distribution.get("p90") is not None:
            if distribution["p10"] < 0.05 and distribution["p90"] > 0.95:
                warnings.append(f"{name} produces very extreme probabilities under transfer stress.")
    return sorted(set(warnings))


def _model() -> Pipeline:
    return Pipeline([
        ("pre", ColumnTransformer([
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), COMMON_FEATURES),
        ])),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def _probability_summary(values: np.ndarray) -> dict[str, float | None]:
    if len(values) == 0:
        return {"mean": None, "p10": None, "p50": None, "p90": None, "min": None, "max": None}
    return {
        "mean": round(float(np.mean(values)), 4),
        "p10": round(float(np.quantile(values, 0.10)), 4),
        "p50": round(float(np.quantile(values, 0.50)), 4),
        "p90": round(float(np.quantile(values, 0.90)), 4),
        "min": round(float(np.min(values)), 4),
        "max": round(float(np.max(values)), 4),
    }


def _psi(source: pd.Series, target: pd.Series, bins: int = 10) -> float | None:
    source = pd.to_numeric(source, errors="coerce").dropna()
    target = pd.to_numeric(target, errors="coerce").dropna()
    if len(source) < bins or len(target) < bins:
        return None
    quantiles = np.unique(np.quantile(source, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        return None
    source_counts, _ = np.histogram(source, bins=quantiles)
    target_counts, _ = np.histogram(target, bins=quantiles)
    source_pct = np.clip(source_counts / max(source_counts.sum(), 1), 1e-6, None)
    target_pct = np.clip(target_counts / max(target_counts.sum(), 1), 1e-6, None)
    return round(float(np.sum((target_pct - source_pct) * np.log(target_pct / source_pct))), 4)


def _extract_baseline_size(value: Any) -> float | None:
    payload = value
    if isinstance(value, str):
        if value.strip() == "":
            return None
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            try:
                payload = ast.literal_eval(value)
            except Exception:
                return None
    if isinstance(payload, dict):
        for key in ("baseline_longest_diameter_mm", "tumor_size", "TUMOR_SIZE"):
            if key in payload:
                try:
                    return float(payload[key])
                except (TypeError, ValueError):
                    return None
    return None


def _safe_auc(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    if len(set(labels.tolist())) < 2:
        return None
    return round(float(roc_auc_score(labels, probabilities)), 4)


def _rounded_mean(values: pd.Series) -> float | None:
    return round(float(values.mean()), 4) if len(values) else None


def _string_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series([""] * len(frame), index=frame.index, dtype=str)
    return frame[column].fillna("").astype(str)


def _empty_common_frame(*, include_label: bool = True) -> pd.DataFrame:
    columns = COMMON_FEATURES + (["label"] if include_label else [])
    return pd.DataFrame(columns=columns)


def _read_frame(path: str | Path) -> pd.DataFrame:
    resolved = _resolve(path)
    if not resolved.exists() or resolved.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(resolved)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate
