"""
Deterministic CBC and symptom risk flags.

Thresholds are loaded from config/safety_thresholds.yaml so they are versioned
and auditable.  The threshold_config_version is embedded in every evidence
payload; audit logs therefore remain reproducible across threshold changes.

CLAIM BOUNDARY: these flags are engineering signals for clinical attention
prompts — not diagnostic criteria or treatment orders.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

# PyYAML is in the standard scientific stack; fall back to hard-coded defaults
# if the file or package is unavailable (keeps unit tests hermetic).
try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

_THRESHOLDS_PATH = Path(__file__).parent.parent.parent / "config" / "safety_thresholds.yaml"


@functools.lru_cache(maxsize=1)
def _load_config() -> dict:
    if _YAML_AVAILABLE and _THRESHOLDS_PATH.exists():
        with open(_THRESHOLDS_PATH, encoding="utf-8") as fh:
            return _yaml.safe_load(fh) or {}
    return {}


def _cfg() -> dict:
    return _load_config()


def _threshold_version() -> str:
    return _cfg().get("threshold_config_version", "v1.1.0")


def _lab_cfg(metric: str) -> dict:
    return (_cfg().get("lab_thresholds") or {}).get(metric, {})


def _rule_cfg(rule: str) -> dict:
    return (_cfg().get("clinical_rules") or {}).get(rule, {})


def _symptom_cfg(key: str) -> dict:
    return (_cfg().get("symptom_thresholds") or {}).get(key, {})


# ── Lab threshold helpers (keep dict for backward compat callers) ─────────────

def _lab_thresholds_dict() -> dict[str, dict]:
    raw = (_cfg().get("lab_thresholds") or {})
    if raw:
        return raw
    # Hard-coded fallback — identical to previous v1.0 values
    return {
        "wbc": {
            "watch": 4.0, "urgent_review": 3.0,
            "unit": "x10^3/uL", "label": "WBC",
            "risk_type": "low_wbc",
            "concern": "possible treatment-related white blood cell suppression",
        },
        "platelets": {
            "watch": 150, "urgent_review": 100,
            "unit": "x10^3/uL", "label": "Platelets",
            "risk_type": "low_platelets", "concern": "possible thrombocytopenia",
        },
        "hemoglobin": {
            "watch": 11.0, "urgent_review": 10.0,
            "unit": "g/dL", "label": "Hemoglobin",
            "risk_type": "low_hemoglobin", "concern": "possible anemia",
        },
    }


# Expose as module-level name for any legacy callers that import it directly.
LAB_THRESHOLDS: dict[str, Any] = _lab_thresholds_dict()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _severity_for_value(value: float, thresholds: dict) -> str | None:
    if value < thresholds["urgent_review"]:
        return "urgent_review"
    if value < thresholds["watch"]:
        return "watch"
    return None


def _risk_message(metric_config: dict, value: float, severity: str) -> str:
    action = (
        "Medical review is recommended."
        if severity == "urgent_review"
        else "Continue monitoring and review in clinical context."
    )
    return (
        f"{metric_config['label']} reached {value} {metric_config['unit']}, "
        f"suggesting {metric_config['concern']}. {action}"
    )


def _base_evidence(**kwargs) -> dict:
    """Attach threshold version to every evidence dict."""
    return {"threshold_config_version": _threshold_version(), **kwargs}


# ── Public API ────────────────────────────────────────────────────────────────

def detect_risks(labs_df):
    risks = []
    thresholds = _lab_thresholds_dict()

    for metric, config in thresholds.items():
        min_index = labs_df[metric].idxmin()
        min_row = labs_df.loc[min_index]
        min_value = float(min_row[metric])
        severity = _severity_for_value(min_value, config)

        if severity:
            risks.append({
                "type": config["risk_type"],
                "category": "lab",
                "severity": severity,
                "message": _risk_message(config, min_value, severity),
                "evidence": _base_evidence(
                    metric=metric,
                    date=str(min_row["date"]),
                    value=min_value,
                    unit=config["unit"],
                    watch_threshold=config["watch"],
                    urgent_review_threshold=config["urgent_review"],
                ),
            })

    return risks


def detect_trend_risk(labs_df):
    risks = []
    wbc_start = float(labs_df["wbc"].iloc[0])
    wbc_min = float(labs_df["wbc"].min())
    if wbc_start <= 0:
        return risks

    drop_percent = ((wbc_start - wbc_min) / wbc_start) * 100
    drop_threshold = float(_rule_cfg("wbc_drop_percent").get("threshold", 50))

    if drop_percent >= drop_threshold:
        risks.append({
            "type": "major_wbc_drop",
            "category": "lab_trend",
            "severity": "urgent_review",
            "message": (
                f"WBC dropped by {drop_percent:.1f}% from baseline. "
                "This should be reviewed with the treatment timeline."
            ),
            "evidence": _base_evidence(
                metric="wbc",
                baseline_value=wbc_start,
                lowest_value=wbc_min,
                percent_drop=round(drop_percent, 1),
                unit=_lab_thresholds_dict()["wbc"]["unit"],
                drop_threshold_pct=drop_threshold,
            ),
        })

    return risks


def detect_symptom_risks(symptoms_df):
    risks = []
    if symptoms_df is None or symptoms_df.empty:
        return risks

    severity_threshold = int(_symptom_cfg("high_severity").get("threshold", 7))
    high_symptoms = symptoms_df[symptoms_df["severity"] >= severity_threshold]
    for _, row in high_symptoms.iterrows():
        risks.append({
            "type": "high_severity_symptom",
            "category": "symptom",
            "severity": "urgent_review",
            "message": (
                f"Patient-reported {row['symptom']} severity was "
                f"{int(row['severity'])}/10. Clinical review may be needed."
            ),
            "evidence": _base_evidence(
                date=str(row["date"]),
                symptom=row["symptom"],
                severity_score=int(row["severity"]),
                severity_threshold=severity_threshold,
                notes=row.get("notes"),
            ),
        })

    return risks


def detect_clinical_rule_risks(labs_df, symptoms_df, treatments_df):
    """Deterministic multi-signal oncology monitoring rules.

    Conservative clinical-support flags — not diagnosis or treatment orders.
    """
    risks = []
    risks.extend(_critical_cbc_rules(labs_df))
    risks.extend(_fever_after_treatment_rules(symptoms_df, treatments_df))
    risks.extend(_cbc_symptom_combination_rules(labs_df, symptoms_df))
    return risks


def _critical_cbc_rules(labs_df):
    if labs_df is None or labs_df.empty:
        return []

    risks = []
    latest = labs_df.sort_values("date").iloc[-1]
    baseline = labs_df.sort_values("date").iloc[0]
    min_wbc = labs_df.loc[labs_df["wbc"].idxmin()]
    min_platelets = labs_df.loc[labs_df["platelets"].idxmin()]
    min_hemoglobin = labs_df.loc[labs_df["hemoglobin"].idxmin()]

    wbc_critical = float(_rule_cfg("critical_wbc_suppression").get("threshold", 2.0))
    plt_critical = float(_rule_cfg("critical_platelet_suppression").get("threshold", 50))
    hgb_critical = float(_rule_cfg("hemoglobin_critical_low").get("threshold", 8.0))
    hgb_drop = float(_rule_cfg("hemoglobin_significant_drop").get("threshold", 2.0))

    if float(min_wbc["wbc"]) < wbc_critical:
        risks.append({
            "type": "critical_wbc_suppression",
            "category": "deterministic_clinical_rule",
            "severity": "urgent_review",
            "message": (
                f"WBC fell below {wbc_critical} x10^3/uL. "
                "Oncology review is recommended, especially during active chemotherapy."
            ),
            "evidence": _base_evidence(
                date=str(min_wbc["date"]),
                metric="wbc",
                value=float(min_wbc["wbc"]),
                threshold=wbc_critical,
            ),
        })

    if float(min_platelets["platelets"]) < plt_critical:
        risks.append({
            "type": "critical_platelet_suppression",
            "category": "deterministic_clinical_rule",
            "severity": "urgent_review",
            "message": (
                f"Platelets fell below {plt_critical} x10^3/uL. "
                "Bleeding-risk review may be needed."
            ),
            "evidence": _base_evidence(
                date=str(min_platelets["date"]),
                metric="platelets",
                value=float(min_platelets["platelets"]),
                threshold=plt_critical,
            ),
        })

    hemoglobin_drop = float(baseline["hemoglobin"]) - float(latest["hemoglobin"])
    if hemoglobin_drop >= hgb_drop or float(min_hemoglobin["hemoglobin"]) < hgb_critical:
        risks.append({
            "type": "clinically_significant_hemoglobin_drop",
            "category": "deterministic_clinical_rule",
            "severity": (
                "urgent_review" if float(min_hemoglobin["hemoglobin"]) < hgb_critical else "watch"
            ),
            "message": (
                f"Hemoglobin changed by {hemoglobin_drop:.1f} g/dL from baseline. "
                "Review anemia trend in the treatment-cycle context."
            ),
            "evidence": _base_evidence(
                baseline_value=float(baseline["hemoglobin"]),
                latest_value=float(latest["hemoglobin"]),
                lowest_value=float(min_hemoglobin["hemoglobin"]),
                latest_date=str(latest["date"]),
                drop_threshold=hgb_drop,
                critical_threshold=hgb_critical,
            ),
        })

    return risks


def _fever_after_treatment_rules(symptoms_df, treatments_df):
    if symptoms_df is None or symptoms_df.empty or treatments_df is None or treatments_df.empty:
        return []

    window_days = int(_rule_cfg("fever_after_chemo_window_days").get("threshold", 14))
    risks = []
    fever_rows = symptoms_df[
        symptoms_df["symptom"].astype(str).str.lower().str.contains(
            "fever|chills", regex=True, na=False
        )
    ]
    if fever_rows.empty:
        return risks

    treatment_dates = list(treatments_df["date"])
    for _, fever in fever_rows.iterrows():
        fever_date = fever["date"]
        recent_cycles = [
            td for td in treatment_dates if 0 <= (fever_date - td).days <= window_days
        ]
        if recent_cycles:
            risks.append({
                "type": "fever_after_recent_chemotherapy",
                "category": "deterministic_clinical_rule",
                "severity": "urgent_review",
                "message": (
                    f"Fever/chills were reported within {window_days} days after treatment. "
                    "This should be reviewed urgently in chemotherapy context."
                ),
                "evidence": _base_evidence(
                    symptom_date=str(fever_date),
                    severity_score=int(fever.get("severity", 0)),
                    recent_treatment_dates=[str(v) for v in recent_cycles],
                    window_days=window_days,
                ),
            })

    return risks


def _cbc_symptom_combination_rules(labs_df, symptoms_df):
    if labs_df is None or labs_df.empty or symptoms_df is None or symptoms_df.empty:
        return []

    combo_window = int(_rule_cfg("cbc_symptom_combo_window_days").get("threshold", 3))
    low_wbc_threshold = float(_rule_cfg("low_wbc_for_combo_rule").get("threshold", 3.0))

    risks = []
    low_wbc_rows = labs_df[labs_df["wbc"] < low_wbc_threshold]
    fever_rows = symptoms_df[
        symptoms_df["symptom"].astype(str).str.lower().str.contains(
            "fever|chills", regex=True, na=False
        )
    ]
    if low_wbc_rows.empty or fever_rows.empty:
        return risks

    for _, fever in fever_rows.iterrows():
        fever_date = fever["date"]
        nearby = low_wbc_rows[
            low_wbc_rows["date"].apply(
                lambda d: abs((d - fever_date).days) <= combo_window
            )
        ]
        if not nearby.empty:
            lowest = nearby.loc[nearby["wbc"].idxmin()]
            risks.append({
                "type": "fever_with_low_wbc",
                "category": "deterministic_clinical_rule",
                "severity": "urgent_review",
                "message": "Fever/chills occurred near a low WBC result. Clinician review should be prioritized.",
                "evidence": _base_evidence(
                    symptom_date=str(fever_date),
                    wbc_date=str(lowest["date"]),
                    wbc=float(lowest["wbc"]),
                    low_wbc_threshold=low_wbc_threshold,
                    combo_window_days=combo_window,
                ),
            })
            break

    return risks
