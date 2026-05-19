from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.services.oncology_canonical_schema import ROOT_DIR


DEFAULT_SYNTHETIC_CSV = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
DEFAULT_ALIGNMENT_PATH = "Data/evals/models/latest_external_distribution_alignment.json"
DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_public_distribution_realism_candidate.json"
DEFAULT_CANDIDATE_CSV = "Data/external_bridge/realism_candidate/temporal_ml_rows_public_realism_candidate.csv"

CLAIM_BOUNDARY = (
    "The public-distribution realism candidate is a simulator-calibration artifact. It adjusts selected synthetic "
    "feature distributions toward public cohort summaries to stress data realism, but it is still synthetic, does "
    "not carry clinician labels, and must not be used as clinical validation or patient-facing prediction evidence."
)


def build_public_distribution_realism_candidate(
    *,
    synthetic_csv: str = DEFAULT_SYNTHETIC_CSV,
    alignment_path: str = DEFAULT_ALIGNMENT_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    candidate_csv: str = DEFAULT_CANDIDATE_CSV,
) -> dict[str, Any]:
    synthetic = _read_frame(synthetic_csv)
    alignment = _read_json(alignment_path)
    candidate = synthetic.copy()

    adjustments: list[dict[str, Any]] = []
    before_after: dict[str, Any] = {}

    age_target = _target_stats(alignment, "age", preferred=("cbioportal_tcga_metabric", "breastdcedl"))
    if "age" in candidate.columns and age_target:
        before = pd.to_numeric(candidate["age"], errors="coerce")
        candidate["age"] = _shift_scale(before, target_mean=age_target["mean"], target_std=age_target.get("std"), low=18, high=95)
        after = pd.to_numeric(candidate["age"], errors="coerce")
        adjustments.append({
            "field": "age",
            "method": "mean_std_shift_scale",
            "target_source": age_target["source"],
            "target_mean": age_target["mean"],
            "target_std": age_target.get("std"),
        })
        before_after["age"] = _gap_summary(before, after, age_target["mean"])

    tumor_target = _target_stats(alignment, "baseline_tumor_size_mm", preferred=("breastdcedl", "cbioportal_tcga_metabric"))
    if "mri_tumor_size_cm" in candidate.columns and tumor_target:
        before_mm = pd.to_numeric(candidate["mri_tumor_size_cm"], errors="coerce") * 10.0
        source_mean = float(before_mm.dropna().mean()) if before_mm.notna().any() else None
        target_mean = tumor_target["mean"]
        factor = float(target_mean / source_mean) if source_mean and source_mean > 0 else 1.0
        # Bound the multiplier so the candidate is a realism stress test, not a new clinical generator.
        bounded_factor = float(np.clip(factor, 0.50, 3.00))
        candidate["mri_tumor_size_cm"] = (pd.to_numeric(candidate["mri_tumor_size_cm"], errors="coerce") * bounded_factor).clip(0.1, 15.0)
        after_mm = pd.to_numeric(candidate["mri_tumor_size_cm"], errors="coerce") * 10.0
        adjustments.append({
            "field": "mri_tumor_size_cm",
            "method": "bounded_scale_to_public_tumor_size_proxy",
            "target_source": tumor_target["source"],
            "target_mean_mm": target_mean,
            "raw_factor": round(factor, 4),
            "bounded_factor": round(bounded_factor, 4),
            "semantic_warning": (
                "Synthetic mri_tumor_size_cm is a monitoring-cycle proxy while BreastDCEDL's "
                "baseline_longest_diameter_mm is a baseline MRI feature; this candidate is for stress testing only."
            ),
        })
        before_after["baseline_tumor_size_mm_proxy"] = _gap_summary(before_mm, after_mm, target_mean)

    candidate_path = _resolve(candidate_csv)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(candidate_path, index=False)

    improvement_count = sum(1 for metrics in before_after.values() if metrics.get("gap_improved"))
    status = "candidate" if improvement_count else "needs_attention"
    payload = {
        "schema_version": "public_distribution_realism_candidate_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "candidate_csv": _display_path(candidate_path),
        "source_csv": _display_path(_resolve(synthetic_csv)),
        "alignment_source": _display_path(_resolve(alignment_path)),
        "rows": int(len(candidate)),
        "patients": int(candidate["patient_id"].nunique()) if "patient_id" in candidate else None,
        "adjustments": adjustments,
        "before_after_gaps": before_after,
        "realism_candidate_decision": {
            "recommendation": "evaluate_separately_do_not_promote",
            "use_for_training": "candidate_ab_test_only",
            "production_replacement_allowed": False,
            "reason": (
                "The candidate improves selected public-distribution gaps but may distort simulator relationships. "
                "It must be compared against the current generator with leakage, calibration, shortcut, and stability gates."
            ),
        },
        "unsupported_claims": [
            "clinical validation",
            "real patient treatment-response prediction",
            "proof that public distributions make the model clinically accurate",
            "treatment recommendation or regimen superiority",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(_resolve(output_path), payload)
    return payload


def _target_stats(alignment: dict[str, Any], field: str, *, preferred: tuple[str, ...]) -> dict[str, Any] | None:
    field_stats = ((alignment.get("numeric_alignment") or {}).get(field) or {})
    for source in preferred:
        stats = field_stats.get(source) or {}
        mean = stats.get("mean")
        if mean is None:
            continue
        p10 = stats.get("p10")
        p90 = stats.get("p90")
        approx_std = None
        if p10 is not None and p90 is not None:
            approx_std = max(float(p90) - float(p10), 0.0) / 2.563
        return {
            "source": source,
            "mean": float(mean),
            "std": round(float(approx_std), 4) if approx_std else None,
        }
    return None


def _shift_scale(values: pd.Series, *, target_mean: float, target_std: float | None, low: float, high: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    source_mean = float(numeric.dropna().mean()) if numeric.notna().any() else target_mean
    source_std = float(numeric.dropna().std(ddof=0)) if numeric.notna().sum() > 1 else None
    if target_std and source_std and source_std > 0:
        adjusted = (numeric - source_mean) * (target_std / source_std) + target_mean
    else:
        adjusted = numeric + (target_mean - source_mean)
    return adjusted.clip(low, high).round(2)


def _gap_summary(before: pd.Series, after: pd.Series, target_mean: float) -> dict[str, Any]:
    before = pd.to_numeric(before, errors="coerce")
    after = pd.to_numeric(after, errors="coerce")
    before_mean = float(before.dropna().mean()) if before.notna().any() else None
    after_mean = float(after.dropna().mean()) if after.notna().any() else None
    before_gap = abs(before_mean - target_mean) if before_mean is not None else None
    after_gap = abs(after_mean - target_mean) if after_mean is not None else None
    return {
        "target_mean": round(float(target_mean), 4),
        "before_mean": round(before_mean, 4) if before_mean is not None else None,
        "after_mean": round(after_mean, 4) if after_mean is not None else None,
        "before_absolute_gap": round(before_gap, 4) if before_gap is not None else None,
        "after_absolute_gap": round(after_gap, 4) if after_gap is not None else None,
        "gap_improved": bool(after_gap is not None and before_gap is not None and after_gap < before_gap),
    }


def _read_frame(path: str | Path) -> pd.DataFrame:
    resolved = _resolve(path)
    if not resolved.exists() or resolved.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(resolved)


def _read_json(path: str | Path) -> dict[str, Any]:
    resolved = _resolve(path)
    if not resolved.exists():
        return {}
    return json.loads(resolved.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return str(path)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate
