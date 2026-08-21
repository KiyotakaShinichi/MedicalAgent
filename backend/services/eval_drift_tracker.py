"""Eval drift tracking.

Reads a small set of headline metrics out of the latest ``Data/evals/*``
JSON artifacts and appends one row per call to
``Data/evals/history/eval_history.jsonl``.  Then computes a drift
report (``eval_history.jsonl`` -> ``latest_eval_drift_report.json``)
that flags regressions in the most recent record relative to the
previous record.

Why on disk and not in a DB
~~~~~~~~~~~~~~~~~~~~~~~~~~~
This is a synthetic-only research project.  JSONL is the right
substrate for now: every commit can review the diff, the file is
small enough to grep, and there is no server lifecycle to manage.
A real deployment would move this to a time-series store.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HISTORY_PATH = Path("Data/evals/history/eval_history.jsonl")
REPORT_PATH = Path("Data/evals/history/latest_eval_drift_report.json")


# ─── Sources of headline metrics ─────────────────────────────────────────────


@dataclass(frozen=True)
class MetricSource:
    """One row in the history file's ``metrics`` dict per source."""
    name: str
    path: Path
    extractor: str  # method name on EvalDriftTracker

    def exists(self) -> bool:
        return self.path.exists()


METRIC_SOURCES: tuple[MetricSource, ...] = (
    MetricSource(
        name="patient_temporal_cv",
        path=Path("Data/evals/models/latest_patient_temporal_cv.json"),
        extractor="_extract_patient_temporal_cv",
    ),
    MetricSource(
        name="adversarial_safety_regression",
        path=Path("Data/evals/safety/latest_adversarial_safety_regression.json"),
        extractor="_extract_adversarial_safety",
    ),
    MetricSource(
        name="adversarial_safety_holdout",
        path=Path("Data/evals/safety/latest_adversarial_safety_holdout.json"),
        extractor="_extract_adversarial_holdout",
    ),
    MetricSource(
        name="uncertainty_aware_retrieval",
        path=Path("Data/evals/rag/latest_uncertainty_aware_retrieval_eval.json"),
        extractor="_extract_uncertainty_retrieval",
    ),
    MetricSource(
        name="emotional_distress",
        path=Path("Data/evals/safety/latest_emotional_distress_eval.json"),
        extractor="_extract_emotional_distress",
    ),
)


# Direction (positive == better) for drift comparison.
METRIC_DIRECTIONS: dict[str, str] = {
    "patient_temporal_cv.auc_mean": "higher_is_better",
    "patient_temporal_cv.auc_optimism_delta": "lower_absolute_is_better",
    "adversarial_safety_regression.overall_attack_block_rate": "higher_is_better",
    "adversarial_safety_regression.urgent_symptom_rate": "higher_is_better",
    "adversarial_safety_regression.negative_control_rate": "higher_is_better",
    "adversarial_safety_holdout.overall_attack_block_rate": "higher_is_better",
    "adversarial_safety_holdout.privacy_pii_rate": "higher_is_better",
    "adversarial_safety_holdout.prompt_injection_rate": "higher_is_better",
    "adversarial_safety_holdout.genetic_risk_misinterpretation_rate": "higher_is_better",
    "adversarial_safety_holdout.vus_misinterpretation_rate": "higher_is_better",
    "uncertainty_aware_retrieval.pass_rate": "higher_is_better",
    "emotional_distress.pass_rate": "higher_is_better",
    "emotional_distress.en_pass_rate": "higher_is_better",
    "emotional_distress.tl_pass_rate": "higher_is_better",
}


def _git_commit_hash() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip()
    except Exception:
        return None


class EvalDriftTracker:
    def __init__(self, *, history_path: Path = HISTORY_PATH, report_path: Path = REPORT_PATH) -> None:
        self.history_path = history_path
        self.report_path = report_path

    # ─── Extraction ──────────────────────────────────────────────────────────

    def _read_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _extract_patient_temporal_cv(self, data: dict[str, Any]) -> dict[str, Any]:
        pat = data.get("patient_level_temporal_cv") or {}
        return {
            "auc_mean": pat.get("roc_auc_mean"),
            "auc_std": pat.get("roc_auc_std"),
            "brier_mean": pat.get("brier_mean"),
            "patient_overlap_pairs": pat.get("patient_overlap_pairs"),
            "auc_optimism_delta": (data.get("headline") or {}).get("auc_optimism_from_naive_cv"),
            "target": data.get("target"),
            "n_folds": data.get("n_folds"),
        }

    def _extract_adversarial_safety(self, data: dict[str, Any]) -> dict[str, Any]:
        by_cat = data.get("by_category") or {}
        return {
            "overall_attack_block_rate": data.get("overall_attack_block_rate"),
            "total_cases": data.get("total_cases"),
            "urgent_symptom_rate": (by_cat.get("urgent_symptom") or {}).get("attack_block_rate"),
            "negative_control_rate": (
                by_cat.get("safe_educational_negative_control") or {}
            ).get("attack_block_rate"),
            "prompt_injection_rate": (by_cat.get("prompt_injection") or {}).get("attack_block_rate"),
            "diagnosis_confirmation_rate": (
                by_cat.get("diagnosis_confirmation") or {}
            ).get("attack_block_rate"),
        }

    def _extract_adversarial_holdout(self, data: dict[str, Any]) -> dict[str, Any]:
        by_cat = data.get("by_category") or {}
        def _rate(name: str) -> float | None:
            block = by_cat.get(name) or {}
            v = block.get("attack_block_rate")
            return float(v) if isinstance(v, (int, float)) else None
        return {
            "overall_attack_block_rate": data.get("overall_attack_block_rate"),
            "total_n": data.get("total_n"),
            "privacy_pii_rate": _rate("privacy_pii"),
            "prompt_injection_rate": _rate("prompt_injection"),
            "genetic_risk_misinterpretation_rate": _rate("genetic_risk_misinterpretation"),
            "vus_misinterpretation_rate": _rate("vus_misinterpretation"),
        }

    def _extract_uncertainty_retrieval(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "pass_rate": data.get("pass_rate"),
            "n_probes": data.get("n_probes"),
        }

    def _extract_emotional_distress(self, data: dict[str, Any]) -> dict[str, Any]:
        by_lang = data.get("by_language") or {}
        return {
            "pass_rate": data.get("pass_rate"),
            "n_probes": data.get("n_probes"),
            "en_pass_rate": (by_lang.get("en") or {}).get("pass_rate"),
            "tl_pass_rate": (by_lang.get("tl") or {}).get("pass_rate"),
        }

    # ─── Append ──────────────────────────────────────────────────────────────

    def build_record(self, *, release_id: str | None = None) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        missing: list[str] = []
        for src in METRIC_SOURCES:
            if not src.exists():
                missing.append(src.name)
                continue
            extractor = getattr(self, src.extractor)
            metrics[src.name] = extractor(self._read_json(src.path))
        return {
            "release_id": release_id or os.environ.get("ONCOTRACK_RELEASE_ID", "dev"),
            "commit_hash": _git_commit_hash(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "missing_sources": missing,
            "metrics": metrics,
        }

    def append_record(self, *, release_id: str | None = None) -> dict[str, Any]:
        record = self.build_record(release_id=release_id)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")
        return record

    # ─── Drift report ────────────────────────────────────────────────────────

    def load_history(self) -> list[dict[str, Any]]:
        if not self.history_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.history_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def build_drift_report(self) -> dict[str, Any]:
        history = self.load_history()
        if not history:
            return {
                "schema_version": "1.0",
                "status": "informational",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "n_records": 0,
                "deltas": [],
                "regressions": [],
                "regression_count": 0,
            }
        latest = history[-1]
        previous = history[-2] if len(history) >= 2 else None

        deltas: list[dict[str, Any]] = []
        regressions: list[dict[str, Any]] = []

        if previous:
            for source, latest_block in (latest.get("metrics") or {}).items():
                prev_block = (previous.get("metrics") or {}).get(source) or {}
                for metric_name, current_value in (latest_block or {}).items():
                    prev_value = prev_block.get(metric_name)
                    if not isinstance(current_value, (int, float)) or not isinstance(prev_value, (int, float)):
                        continue
                    delta = float(current_value) - float(prev_value)
                    full_key = f"{source}.{metric_name}"
                    direction = METRIC_DIRECTIONS.get(full_key)
                    is_regression = self._is_regression(delta, direction)
                    entry = {
                        "metric": full_key,
                        "previous": prev_value,
                        "latest": current_value,
                        "delta": delta,
                        "direction": direction,
                        "is_regression": is_regression,
                    }
                    deltas.append(entry)
                    if is_regression:
                        regressions.append(entry)

        return {
            "schema_version": "1.0",
            "status": "informational",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_records": len(history),
            "latest_release_id": latest.get("release_id"),
            "latest_commit_hash": latest.get("commit_hash"),
            "previous_release_id": previous.get("release_id") if previous else None,
            "deltas": deltas,
            "regressions": regressions,
            "regression_count": len(regressions),
        }

    @staticmethod
    def _is_regression(delta: float, direction: str | None) -> bool:
        # 1% threshold so noise floor doesn't dominate.
        threshold = 0.01
        if direction == "higher_is_better":
            return delta < -threshold
        if direction == "lower_is_better":
            return delta > threshold
        if direction == "lower_absolute_is_better":
            return abs(delta) > threshold
        return False  # No declared direction → never a regression.

    def write_drift_report(self) -> dict[str, Any]:
        report = self.build_drift_report()
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


__all__ = [
    "EvalDriftTracker",
    "HISTORY_PATH",
    "METRIC_SOURCES",
    "METRIC_DIRECTIONS",
    "REPORT_PATH",
]
