from __future__ import annotations

import ast
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backend.services.oncology_canonical_schema import ROOT_DIR


DEFAULT_SYNTHETIC_CSV = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
DEFAULT_BREASTDCEDL_CSV = "Data/external_bridge/canonical_breastdcedl_spy1.csv"
DEFAULT_CBIOPORTAL_CSV = "Data/external_bridge/cbioportal/canonical_cbioportal_breast_rows.csv"
DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_external_distribution_alignment.json"

CLAIM_BOUNDARY = (
    "External distribution alignment compares broad field distributions across synthetic rows and public cohorts. "
    "It is a realism/debugging artifact only, not validation of clinical prediction or treatment utility."
)


def build_external_distribution_alignment(
    *,
    synthetic_csv: str = DEFAULT_SYNTHETIC_CSV,
    breastdcedl_csv: str = DEFAULT_BREASTDCEDL_CSV,
    cbioportal_csv: str = DEFAULT_CBIOPORTAL_CSV,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    synthetic = _synthetic_patient_level(_read_frame(synthetic_csv))
    breastdcedl = _canonical_frame(_read_frame(breastdcedl_csv))
    cbio = _canonical_frame(_read_frame(cbioportal_csv))

    cohorts = {
        "synthetic": synthetic,
        "breastdcedl": breastdcedl,
        "cbioportal_tcga_metabric": cbio,
    }
    payload = {
        "schema_version": "external_distribution_alignment_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong" if len(cbio) > 0 else "needs_attention",
        "cohort_sizes": {name: int(len(frame)) for name, frame in cohorts.items()},
        "numeric_alignment": {
            "age": _numeric_summary(cohorts, "age"),
            "baseline_tumor_size_mm": _numeric_summary(cohorts, "baseline_tumor_size_mm"),
        },
        "categorical_alignment": {
            "molecular_subtype": _categorical_summary(cohorts, "molecular_subtype"),
            "stage": _categorical_summary(cohorts, "stage"),
            "er_status": _categorical_summary(cohorts, "er_status"),
            "pr_status": _categorical_summary(cohorts, "pr_status"),
            "her2_status": _categorical_summary(cohorts, "her2_status"),
        },
        "treatment_context_alignment": _treatment_alignment(cohorts),
        "largest_gaps": _largest_gaps(cohorts),
        "recommended_actions": [
            "Do not tune the model to match public cohorts blindly; first check target compatibility.",
            "Use tumor-size and subtype gaps to tune simulator realism only as a separate candidate dataset.",
            "Keep cBioPortal survival/progression labels separate from OncoTrack response-score labels.",
            "Use cBioPortal rows for distribution and schema checks, not patient-facing predictions.",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    output = _resolve(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _synthetic_patient_level(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    final = frame.sort_values(["patient_id", "cycle"]).groupby("patient_id", as_index=False).tail(1).copy()
    subtype = final["molecular_subtype"].astype(str)
    final["baseline_tumor_size_mm"] = pd.to_numeric(final["mri_tumor_size_cm"], errors="coerce") * 10.0
    final["er_status"] = subtype.str.contains("HR\\+", case=False, regex=True).map({True: "positive", False: "unknown"})
    final["pr_status"] = "unknown"
    final["her2_status"] = subtype.str.contains("HER2\\+", case=False, regex=True).map({True: "positive", False: "negative"})
    final.loc[subtype.str.contains("triple", case=False, regex=False), ["er_status", "pr_status", "her2_status"]] = "negative"
    final["treatment_modalities"] = final["regimen"].astype(str).map(_synthetic_modalities)
    return final


def _canonical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    for column in ("age", "baseline_tumor_size_mm"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "imaging_features" in frame.columns and "baseline_tumor_size_mm" not in frame.columns:
        frame["baseline_tumor_size_mm"] = frame["imaging_features"].map(_extract_baseline_size)
    if "treatment_modalities" in frame.columns:
        frame["treatment_modalities"] = frame["treatment_modalities"].map(_parse_listish)
    else:
        frame["treatment_modalities"] = [[] for _ in range(len(frame))]
    return frame


def _numeric_summary(cohorts: dict[str, pd.DataFrame], column: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, frame in cohorts.items():
        values = pd.to_numeric(frame[column], errors="coerce").dropna() if column in frame else pd.Series(dtype=float)
        out[name] = {
            "available": int(len(values)),
            "missing_rate": round(1.0 - (len(values) / max(len(frame), 1)), 4),
            "mean": round(float(values.mean()), 4) if len(values) else None,
            "median": round(float(values.median()), 4) if len(values) else None,
            "p10": round(float(values.quantile(0.10)), 4) if len(values) else None,
            "p90": round(float(values.quantile(0.90)), 4) if len(values) else None,
        }
    return out


def _categorical_summary(cohorts: dict[str, pd.DataFrame], column: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, frame in cohorts.items():
        if column not in frame:
            out[name] = {"available": 0, "distribution": {}}
            continue
        values = frame[column].fillna("unknown").astype(str)
        counts = Counter(value if value else "unknown" for value in values)
        total = max(sum(counts.values()), 1)
        out[name] = {
            "available": int(len(values)),
            "distribution": {
                key: round(count / total, 4)
                for key, count in counts.most_common(12)
            },
        }
    return out


def _treatment_alignment(cohorts: dict[str, pd.DataFrame]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, frame in cohorts.items():
        counts: Counter[str] = Counter()
        for value in frame.get("treatment_modalities", []):
            for item in _parse_listish(value):
                counts[str(item)] += 1
        total = max(len(frame), 1)
        out[name] = {
            "rates": {key: round(count / total, 4) for key, count in counts.most_common()},
            "note": "Treatment context is source-specific and not comparable as treatment efficacy.",
        }
    return out


def _largest_gaps(cohorts: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    synthetic = cohorts.get("synthetic", pd.DataFrame())
    for cohort_name in ("breastdcedl", "cbioportal_tcga_metabric"):
        other = cohorts.get(cohort_name, pd.DataFrame())
        for column in ("age", "baseline_tumor_size_mm"):
            if column in synthetic and column in other:
                s = pd.to_numeric(synthetic[column], errors="coerce").dropna()
                o = pd.to_numeric(other[column], errors="coerce").dropna()
                if len(s) and len(o):
                    gaps.append({
                        "cohort": cohort_name,
                        "field": column,
                        "synthetic_mean": round(float(s.mean()), 4),
                        "external_mean": round(float(o.mean()), 4),
                        "absolute_mean_delta": round(abs(float(s.mean() - o.mean())), 4),
                    })
    gaps.sort(key=lambda item: item["absolute_mean_delta"], reverse=True)
    return gaps[:10]


def _read_frame(path: str) -> pd.DataFrame:
    resolved = _resolve(path)
    if not resolved.exists() or resolved.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(resolved)


def _extract_baseline_size(value: Any) -> float | None:
    payload = value
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return None
    if isinstance(payload, dict):
        for key in ("baseline_longest_diameter_mm", "tumor_size", "TUMOR_SIZE"):
            if key in payload:
                try:
                    return float(payload[key])
                except (TypeError, ValueError):
                    return None
    return None


def _parse_listish(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in {None, ""}:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(value)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return [part for part in value.split("+") if part and part != "unknown"]
    return []


def _synthetic_modalities(regimen: str) -> list[str]:
    lower = regimen.lower()
    modalities = ["chemotherapy"] if any(term in lower for term in ("ac", "paclitaxel", "carboplatin", "tchp")) else []
    if "tchp" in lower:
        modalities.append("targeted_anti_her2")
    if "endocrine" in lower:
        modalities.append("endocrine")
    return modalities


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate
