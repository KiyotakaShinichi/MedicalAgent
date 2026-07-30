"""Paired statistical evidence for frozen RAG baseline comparisons.

The baseline comparison already evaluates every configuration on the same
cases. This module uses that pairing to quantify uncertainty without changing
retrieval, the goldset, aliases, or source-governance policy.
"""
from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from backend.services.oncology_canonical_schema import ROOT_DIR


DEFAULT_INPUT_PATH = Path("Data/evals/rag/latest_rag_baseline_comparison.json")
DEFAULT_OUTPUT_PATH = Path(
    "Data/evals/rag/latest_rag_paired_statistical_comparison.json"
)
DEFAULT_DOC_PATH = Path("docs/evals/rag_paired_statistical_comparison.md")

BOOTSTRAP_REPLICATES = 4000
PERMUTATION_REPLICATES = 8000
SEED = 20260730

METRICS: dict[str, dict[str, Any]] = {
    "recall_at_10": {"higher_is_better": True, "practical_delta": 0.02},
    "mrr": {"higher_is_better": True, "practical_delta": 0.02},
    "ndcg_at_10": {"higher_is_better": True, "practical_delta": 0.02},
    "citation_precision": {"higher_is_better": True, "practical_delta": 0.02},
    "claim_supported": {"higher_is_better": True, "practical_delta": 0.02},
    "unsupported_context": {"higher_is_better": False, "practical_delta": 0.02},
    "refusal_correct": {"higher_is_better": True, "practical_delta": 0.01},
    "source_tier_correct": {"higher_is_better": True, "practical_delta": 0.02},
    "latency_ms": {"higher_is_better": False, "practical_delta": 25.0},
}

COMPARISONS = (
    {
        "id": "raw_hybrid_rewrite_vs_bm25",
        "baseline": "bm25_only",
        "candidate": "hybrid_rrf_query_rewrite",
        "scope": "raw retrieval ablation; not the full governed stack",
    },
    {
        "id": "full_governed_stack_vs_bm25",
        "baseline": "bm25_only",
        "candidate": "hybrid_rrf_query_rewrite_parent_child_source_tier",
        "scope": "full source-governed stack versus sparse baseline",
    },
    {
        "id": "experimental_pruner_vs_full_governed_stack",
        "baseline": "hybrid_rrf_query_rewrite_parent_child_source_tier",
        "candidate": "hybrid_rrf_query_rewrite_parent_child_source_tier_pruned",
        "scope": "experimental pruner ablation; not a live-route promotion",
    },
)

CLAIM_BOUNDARY = (
    "This is paired statistical evidence over one internally authored frozen "
    "engineering goldset. It does not establish clinical validation, external "
    "generalisation, patient benefit, or production healthcare readiness."
)


def build_rag_paired_statistical_comparison(
    *,
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    doc_path: str | Path = DEFAULT_DOC_PATH,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    permutation_replicates: int = PERMUTATION_REPLICATES,
    seed: int = SEED,
) -> dict[str, Any]:
    source = _read_json(input_path)
    configurations = source.get("configurations") or {}
    results = []
    for index, comparison in enumerate(COMPARISONS):
        baseline = _configuration(configurations, comparison["baseline"])
        candidate = _configuration(configurations, comparison["candidate"])
        result = _compare(
            comparison,
            baseline,
            candidate,
            bootstrap_replicates=bootstrap_replicates,
            permutation_replicates=permutation_replicates,
            seed=seed + index * 1000,
        )
        results.append(result)

    full = next(
        item for item in results if item["id"] == "full_governed_stack_vs_bm25"
    )
    payload = {
        "schema_version": "rag_paired_statistical_comparison_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "acceptable",
        "clinical_validation": False,
        "external_validation": False,
        "goldset_path": source.get("goldset_path"),
        "goldset_case_count": source.get("total_n"),
        "goldset_was_used_for_tuning": source.get("was_used_for_tuning"),
        "internal_vs_external_authored": source.get("internal_vs_external_authored"),
        "method": {
            "paired_case_alignment": True,
            "bootstrap_replicates": bootstrap_replicates,
            "permutation_replicates": permutation_replicates,
            "confidence_level": 0.95,
            "multiple_comparison_correction": "Holm within each configuration comparison",
            "random_seed": seed,
            "primary_metric": "recall_at_10",
        },
        "comparisons": results,
        "headline": {
            "full_stack_improvement_proven_vs_bm25": full["improvement_proven"],
            "full_stack_recall_at_10_favorable_delta": (
                full["metrics"]["recall_at_10"]["favorable_delta"]
            ),
            "full_stack_recall_at_10_ci95": (
                full["metrics"]["recall_at_10"]["favorable_delta_ci95"]
            ),
            "full_stack_recall_at_10_adjusted_p_value": (
                full["metrics"]["recall_at_10"]["adjusted_p_value"]
            ),
            "interpretation": (
                "A negative favorable delta means the governed stack trails BM25 "
                "on raw recall. Governance gains must be reported separately."
            ),
        },
        "limitations": [
            "The goldset is internally authored and has only one case sample.",
            "Bootstrap and randomisation quantify uncertainty inside this goldset only.",
            "Gold source labels and alias normalisation remain judgment-dependent.",
            "Latency is local engineering timing, not cloud or production traffic.",
            "Statistical significance does not imply clinical or practical value.",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(output_path, payload)
    _write_doc(doc_path, payload)
    return payload


def _compare(
    comparison: dict[str, str],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    bootstrap_replicates: int,
    permutation_replicates: int,
    seed: int,
) -> dict[str, Any]:
    baseline_cases = {
        str(row.get("case_id")): row for row in baseline.get("cases") or []
    }
    candidate_cases = {
        str(row.get("case_id")): row for row in candidate.get("cases") or []
    }
    case_ids = sorted(set(baseline_cases) & set(candidate_cases))
    metric_results: dict[str, dict[str, Any]] = {}
    raw_p_values: dict[str, float] = {}
    for offset, (metric, contract) in enumerate(METRICS.items()):
        baseline_values = [_numeric(baseline_cases[item].get(metric)) for item in case_ids]
        candidate_values = [_numeric(candidate_cases[item].get(metric)) for item in case_ids]
        pairs = [
            (left, right)
            for left, right in zip(baseline_values, candidate_values)
            if left is not None and right is not None
        ]
        result = _paired_metric(
            pairs,
            higher_is_better=bool(contract["higher_is_better"]),
            practical_delta=float(contract["practical_delta"]),
            bootstrap_replicates=bootstrap_replicates,
            permutation_replicates=permutation_replicates,
            seed=seed + offset,
        )
        metric_results[metric] = result
        raw_p_values[metric] = float(result["raw_p_value"])

    adjusted = _holm_adjust(raw_p_values)
    for metric, adjusted_p in adjusted.items():
        row = metric_results[metric]
        row["adjusted_p_value"] = adjusted_p
        row["statistically_significant_after_correction"] = (
            adjusted_p <= 0.05
            and not (
                row["favorable_delta_ci95"][0]
                <= 0
                <= row["favorable_delta_ci95"][1]
            )
        )

    primary = metric_results["recall_at_10"]
    improvement_proven = bool(
        primary["favorable_delta"] >= primary["practical_delta"]
        and primary["favorable_delta_ci95"][0] > 0
        and primary["adjusted_p_value"] <= 0.05
    )
    return {
        **comparison,
        "paired_case_count": len(case_ids),
        "missing_from_baseline": sorted(set(candidate_cases) - set(baseline_cases)),
        "missing_from_candidate": sorted(set(baseline_cases) - set(candidate_cases)),
        "metrics": metric_results,
        "improvement_proven": improvement_proven,
        "decision": "evidence_supports_primary_lift" if improvement_proven else "not_proven",
    }


def _paired_metric(
    pairs: list[tuple[float, float]],
    *,
    higher_is_better: bool,
    practical_delta: float,
    bootstrap_replicates: int,
    permutation_replicates: int,
    seed: int,
) -> dict[str, Any]:
    if not pairs:
        return {
            "n": 0,
            "baseline_mean": None,
            "candidate_mean": None,
            "raw_delta": None,
            "favorable_delta": None,
            "favorable_delta_ci95": [None, None],
            "raw_p_value": 1.0,
            "standardized_paired_effect": None,
            "practical_delta": practical_delta,
        }
    baseline = [item[0] for item in pairs]
    candidate = [item[1] for item in pairs]
    raw_differences = [right - left for left, right in pairs]
    favorable = raw_differences if higher_is_better else [-value for value in raw_differences]
    observed = mean(favorable)
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(max(200, bootstrap_replicates)):
        sampled = [favorable[rng.randrange(len(favorable))] for _ in favorable]
        bootstrap.append(mean(sampled))
    bootstrap.sort()
    ci = [
        round(_quantile(bootstrap, 0.025), 6),
        round(_quantile(bootstrap, 0.975), 6),
    ]
    extreme = 0
    observed_abs = abs(observed)
    for _ in range(max(500, permutation_replicates)):
        permuted = mean(
            value if rng.random() >= 0.5 else -value for value in favorable
        )
        if abs(permuted) >= observed_abs - 1e-12:
            extreme += 1
    p_value = (extreme + 1) / (max(500, permutation_replicates) + 1)
    spread = pstdev(favorable)
    effect = observed / spread if spread > 1e-12 else (math.inf if observed else 0.0)
    return {
        "n": len(pairs),
        "baseline_mean": round(mean(baseline), 6),
        "candidate_mean": round(mean(candidate), 6),
        "raw_delta": round(mean(raw_differences), 6),
        "favorable_delta": round(observed, 6),
        "favorable_delta_ci95": ci,
        "raw_p_value": round(p_value, 6),
        "standardized_paired_effect": (
            round(effect, 6) if math.isfinite(effect) else "infinite_zero_variance"
        ),
        "practical_delta": practical_delta,
        "practically_meaningful": observed >= practical_delta,
        "direction": "higher_is_better" if higher_is_better else "lower_is_better",
    }


def _holm_adjust(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: item[1])
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        corrected = min(1.0, (total - index) * value)
        running = max(running, corrected)
        adjusted[name] = round(running, 6)
    return adjusted


def _configuration(configurations: Any, name: str) -> dict[str, Any]:
    if isinstance(configurations, dict):
        value = configurations.get(name)
        if isinstance(value, dict):
            return value
    raise ValueError(f"Missing RAG configuration: {name}")


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    position = (len(values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def _read_json(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    full = candidate if candidate.is_absolute() else ROOT_DIR / candidate
    payload = json.loads(full.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {full}")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    candidate = Path(path)
    full = candidate if candidate.is_absolute() else ROOT_DIR / candidate
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_doc(path: str | Path, payload: dict[str, Any]) -> None:
    candidate = Path(path)
    full = candidate if candidate.is_absolute() else ROOT_DIR / candidate
    full.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Paired RAG Statistical Comparison",
        "",
        "This report uses the case pairing already present in the frozen internal "
        "RAG baseline comparison. It does not change retrieval or tune the goldset.",
        "",
    ]
    for comparison in payload["comparisons"]:
        recall = comparison["metrics"]["recall_at_10"]
        lines.extend(
            [
                f"## {comparison['id']}",
                "",
                f"- Scope: {comparison['scope']}",
                f"- Paired cases: `{comparison['paired_case_count']}`",
                f"- Favorable Recall@10 delta: `{recall['favorable_delta']}`",
                f"- 95% paired bootstrap CI: `{recall['favorable_delta_ci95']}`",
                f"- Holm-adjusted p-value: `{recall['adjusted_p_value']}`",
                f"- Improvement proven: `{comparison['improvement_proven']}`",
                "",
            ]
        )
    lines.extend(["## Boundary", "", payload["claim_boundary"], ""])
    full.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "build_rag_paired_statistical_comparison",
    "_holm_adjust",
]
