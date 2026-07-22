"""Build an isolated I-SPY2/TCIA tabular stress benchmark.

This bridge never feeds NLCare training or patient-facing inference. The pCR
label is a different task from NLCare's synthetic monitoring heads, so results
are external engineering stress evidence only.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
BRIDGE_DIR = ROOT / "Data" / "external_bridge" / "ispy2_tcia"
CLINICAL_PATH = BRIDGE_DIR / "ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx"
MRI_PATH = BRIDGE_DIR / "Multi-feature-MRI-NACT-Data.xlsx"
CANONICAL_PATH = BRIDGE_DIR / "canonical_ispy2_tabular.csv"
OUTPUT_PATH = ROOT / "Data" / "evals" / "models" / "latest_ispy2_tcia_external_stress.json"

EXPECTED_SHA256 = {
    CLINICAL_PATH.name: "c016962d2d1e23686746ad3e74a58caeb2d1362f6393fd6209c10723f87c3a53",
    MRI_PATH.name: "f714c7784b1e57daa74d7cfb20db71cd432b4e4596b9b4eacdd5a76b7f8a58dc",
}
OFFICIAL_COLLECTION_URL = "https://www.cancerimagingarchive.net/collection/ispy2/"
SEEDS = (17, 29, 43, 71, 101)

CLINICAL_REQUIRED = {
    "Patient_ID", "Arm", "HR", "HER2", "MP", "pCR", "Age_at_Screening",
    "Race", "menopausal_status", "ethnicity",
}
MRI_REQUIRED = {
    "CLINICAL-TRIAL-SUBJECT-ID", "VOLUME_TUM_BLU_V10", "SPHERICITY_T0",
    "LD_T0", "BPE_5slice_mean_T0", "FTV_pch_T0_T1",
    "Sphericity_pch_T0_T1", "LD_pch_T0_T1", "BPE_pch_T0_T1",
}

CLINICAL_FEATURES = ["age_at_screening", "hr", "her2", "mp"]
BASELINE_MRI_FEATURES = CLINICAL_FEATURES + [
    "volume_tum_blu_v10", "sphericity_t0", "ld_t0", "bpe_5slice_mean_t0",
]
EARLY_CHANGE_FEATURES = BASELINE_MRI_FEATURES + [
    "ftv_pch_t0_t1", "sphericity_pch_t0_t1", "ld_pch_t0_t1", "bpe_pch_t0_t1",
]
FEATURE_SETS = {
    "clinical_only": CLINICAL_FEATURES,
    "baseline_mri": BASELINE_MRI_FEATURES,
    "early_change": EARLY_CHANGE_FEATURES,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_inputs(clinical_path: Path, mri_path: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in (clinical_path, mri_path):
        if not path.exists():
            raise FileNotFoundError(f"Required official I-SPY2 file is missing: {path}")
        actual = _sha256(path)
        expected = EXPECTED_SHA256.get(path.name)
        if expected is None or actual != expected:
            raise ValueError(f"Checksum mismatch for {path.name}: {actual}")
        hashes[path.name] = actual
    return hashes


def _require_columns(frame: pd.DataFrame, expected: set[str], label: str) -> None:
    missing = sorted(expected - set(frame.columns))
    if missing:
        raise ValueError(f"{label} schema missing required columns: {missing}")


def _case_key(value: Any) -> str:
    return hashlib.sha256(f"ispy2:{int(value)}".encode("utf-8")).hexdigest()[:16]


def build_canonical_frame(
    clinical_path: Path = CLINICAL_PATH,
    mri_path: Path = MRI_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    hashes = _verify_inputs(clinical_path, mri_path)
    clinical = pd.read_excel(clinical_path, sheet_name="ISPY2_n985_TCIA_clinical")
    mri = pd.read_excel(mri_path, sheet_name="datawith4visits")
    _require_columns(clinical, CLINICAL_REQUIRED, "clinical")
    _require_columns(mri, MRI_REQUIRED, "MRI")

    joined = clinical.merge(
        mri,
        left_on="Patient_ID",
        right_on="CLINICAL-TRIAL-SUBJECT-ID",
        how="inner",
        validate="one_to_one",
    )
    canonical = pd.DataFrame({
        "external_case_key": joined["Patient_ID"].map(_case_key),
        "external_target_pcr": pd.to_numeric(joined["pCR"], errors="coerce"),
        "age_at_screening": pd.to_numeric(joined["Age_at_Screening"], errors="coerce"),
        "hr": pd.to_numeric(joined["HR"], errors="coerce"),
        "her2": pd.to_numeric(joined["HER2"], errors="coerce"),
        "mp": pd.to_numeric(joined["MP"], errors="coerce"),
        "race": joined["Race"].fillna("unknown").astype(str),
        "menopausal_status": joined["menopausal_status"].fillna("unknown").astype(str),
        "ethnicity": joined["ethnicity"].fillna("unknown").astype(str),
    })
    for column in mri.columns:
        if column == "CLINICAL-TRIAL-SUBJECT-ID":
            continue
        canonical[column.lower()] = pd.to_numeric(joined[column], errors="coerce")
    canonical = canonical.dropna(subset=["external_target_pcr"]).reset_index(drop=True)
    canonical["external_target_pcr"] = canonical["external_target_pcr"].astype(int)

    manifest = {
        "input_sha256": hashes,
        "clinical_row_count": int(len(clinical)),
        "mri_row_count": int(len(mri)),
        "joined_row_count": int(len(canonical)),
        "raw_subject_identifier_exported": False,
        "treatment_arm_exported_or_used_as_feature": False,
    }
    return canonical, manifest


def _models(seed: int) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        ]),
        "gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", GradientBoostingClassifier(random_state=seed, n_estimators=100, max_depth=2)),
        ]),
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    keys = sorted({(row["feature_set"], row["model"]) for row in rows})
    for feature_set, model in keys:
        group = [row for row in rows if row["feature_set"] == feature_set and row["model"] == model]
        item: dict[str, Any] = {"feature_set": feature_set, "model": model, "n_seeds": len(group)}
        for metric in ("auroc", "average_precision", "brier"):
            values = [float(row[metric]) for row in group]
            item[f"mean_{metric}"] = round(float(np.mean(values)), 6)
            item[f"min_{metric}"] = round(float(np.min(values)), 6)
            item[f"max_{metric}"] = round(float(np.max(values)), 6)
        result.append(item)
    return result


def _bootstrap_mean_ci(values: list[float], seed: int = 20260722) -> list[float] | None:
    if not values:
        return None
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    means = [float(rng.choice(array, size=len(array), replace=True).mean()) for _ in range(2000)]
    return [round(float(np.quantile(means, 0.025)), 6), round(float(np.quantile(means, 0.975)), 6)]


def _paired_feature_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for model in ("logistic_regression", "gradient_boosting"):
        clinical = {row["seed"]: row for row in rows if row["model"] == model and row["feature_set"] == "clinical_only"}
        early = {row["seed"]: row for row in rows if row["model"] == model and row["feature_set"] == "early_change"}
        for metric in ("auroc", "average_precision", "brier"):
            deltas = [float(early[seed][metric]) - float(clinical[seed][metric]) for seed in SEEDS]
            output.append({
                "model": model,
                "comparison": "early_change_minus_clinical_only",
                "metric": metric,
                "mean_delta": round(float(np.mean(deltas)), 6),
                "bootstrap_95_ci": _bootstrap_mean_ci(deltas),
                "n_paired_seeds": len(deltas),
                "direction_note": "lower_is_better" if metric == "brier" else "higher_is_better",
            })
    return output


def run_ispy2_tcia_external_stress(
    clinical_path: Path = CLINICAL_PATH,
    mri_path: Path = MRI_PATH,
    canonical_path: Path = CANONICAL_PATH,
    output_path: Path = OUTPUT_PATH,
) -> dict[str, Any]:
    canonical, input_manifest = build_canonical_frame(clinical_path, mri_path)
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(canonical_path, index=False)

    rows: list[dict[str, Any]] = []
    target = canonical["external_target_pcr"]
    indices = np.arange(len(canonical))
    for seed in SEEDS:
        train_idx, test_idx = train_test_split(
            indices, test_size=0.25, random_state=seed, stratify=target,
        )
        for feature_set, columns in FEATURE_SETS.items():
            x_train = canonical.loc[train_idx, columns]
            x_test = canonical.loc[test_idx, columns]
            y_train = target.iloc[train_idx]
            y_test = target.iloc[test_idx]
            for model_name, model in _models(seed).items():
                model.fit(x_train, y_train)
                probability = model.predict_proba(x_test)[:, 1]
                rows.append({
                    "seed": seed,
                    "feature_set": feature_set,
                    "model": model_name,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "auroc": round(float(roc_auc_score(y_test, probability)), 6),
                    "average_precision": round(float(average_precision_score(y_test, probability)), 6),
                    "brier": round(float(brier_score_loss(y_test, probability)), 6),
                })

    payload = {
        "schema_version": "ispy2_tcia_external_stress_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "acceptable_external_engineering_stress",
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "used_for_nlcare_training": False,
        "patient_facing_allowed": False,
        "promotion_allowed": False,
        "source": {
            "dataset": "I-SPY2 Imaging Cohort 1",
            "publisher": "The Cancer Imaging Archive",
            "official_collection_url": OFFICIAL_COLLECTION_URL,
            "license": "CC BY 4.0 as stated by the official collection page",
            **input_manifest,
        },
        "canonical_export": {
            "path": canonical_path.relative_to(ROOT).as_posix() if canonical_path.is_relative_to(ROOT) else str(canonical_path),
            "sha256": _sha256(canonical_path),
            "row_count": int(len(canonical)),
            "raw_subject_identifier_exported": False,
            "treatment_arm_exported_or_used_as_feature": False,
        },
        "task_boundary": {
            "external_target": "pathologic complete response (pCR)",
            "nlcare_target_match": False,
            "reason": "pCR is not the same endpoint as NLCare's synthetic longitudinal monitoring heads.",
            "allowed_reading": "separate public-data pipeline and model stress benchmark",
        },
        "protocol": {
            "seeds": list(SEEDS),
            "split": "repeated stratified 75/25 holdout",
            "feature_sets": FEATURE_SETS,
            "treatment_arm_used": False,
            "models": ["logistic_regression", "gradient_boosting"],
        },
        "per_seed_results": rows,
        "aggregate_results": _aggregate(rows),
        "paired_feature_deltas": _paired_feature_deltas(rows),
        "claim_boundary": (
            "Public-data engineering stress evidence only. This is not clinical validation, "
            "not an NLCare model promotion result, and not evidence of patient benefit or safety."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload

