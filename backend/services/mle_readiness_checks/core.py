import hashlib
import json
from pathlib import Path

import pandas as pd


def check(name, category, status, value, threshold, meaning, hard_gate, remediation):
    return {
        "name": name,
        "category": category,
        "status": status,
        "value": value,
        "threshold": threshold,
        "meaning": meaning,
        "hard_gate": bool(hard_gate),
        "remediation": remediation,
    }


def overall_status(checks):
    hard_failed = any(item["hard_gate"] and item["status"] == "failed" for item in checks)
    if hard_failed:
        return "failed"
    statuses = [item["status"] for item in checks if item["status"] != "unavailable"]
    if any(status == "failed" for status in statuses):
        return "unideal"
    if any(status == "unideal" for status in statuses):
        return "unideal"
    if any(status == "acceptable" for status in statuses):
        return "acceptable"
    if statuses and all(status in {"passed", "strong"} for status in statuses):
        return "strong"
    return "unavailable"


def release_recommendation(checks):
    status = overall_status(checks)
    if status == "failed":
        return "do_not_promote_hard_gate_failed"
    if status == "unideal":
        return "candidate_only_fix_calibration_or_slice_gaps_before_strong_claims"
    if status == "acceptable":
        return "acceptable_for_poc_demo_with_limitations"
    if status == "strong":
        return "strong_for_engineering_poc_not_clinical_validation"
    return "insufficient_artifacts"


def poc_demo_readiness(checks, category_statuses, hard_failures):
    if hard_failures:
        return {
            "status": "not_ready",
            "recommendation": "do_not_demo_until_hard_gates_pass",
            "reason": "One or more hard gates failed.",
            "required_categories": ["artifacts", "data_contract", "safety_regression"],
            "blocking_categories": sorted({item["category"] for item in hard_failures}),
            "advisory_gaps": advisory_gaps(checks),
            "claim_boundary": "Not suitable for clinical use or unsupervised patient decision-making.",
        }

    required_categories = ["artifacts", "data_contract", "safety_regression"]
    acceptable_statuses = {"strong", "passed", "acceptable"}
    blocking_categories = [
        category
        for category in required_categories
        if category_statuses.get(category) not in acceptable_statuses
    ]
    feature_store_status = category_statuses.get("feature_store")
    if feature_store_status in {"failed", "unavailable"}:
        blocking_categories.append("feature_store")

    status = "ready_with_limitations" if not blocking_categories else "needs_poc_fixes"
    recommendation = (
        "ok_for_supervised_engineering_demo_with_disclaimers"
        if status == "ready_with_limitations"
        else "fix_blocking_poc_gates_before_demo"
    )
    reason = (
        "Core artifacts, data contract, and agent safety regression gates are usable for a supervised PoC demo."
        if status == "ready_with_limitations"
        else "One or more PoC-required engineering gates are not yet acceptable."
    )
    return {
        "status": status,
        "recommendation": recommendation,
        "reason": reason,
        "required_categories": required_categories,
        "blocking_categories": sorted(set(blocking_categories)),
        "advisory_gaps": advisory_gaps(checks),
        "claim_boundary": (
            "PoC/demo readiness means the engineering workflow is demonstrable with synthetic/demo data. "
            "It is not production readiness, clinical validation, or permission to make diagnosis/treatment claims."
        ),
    }


def advisory_gaps(checks):
    return [
        {
            "check": item["name"],
            "category": item["category"],
            "status": item["status"],
            "remediation": item["remediation"],
        }
        for item in checks
        if not item["hard_gate"] and item["status"] in {"unideal", "unavailable", "failed"}
    ][:10]


def category_statuses(checks):
    categories: dict[str, list[str]] = {}
    for item in checks:
        categories.setdefault(item["category"], []).append(item["status"])
    return {category: worst_status(statuses) for category, statuses in categories.items()}


def next_actions(checks):
    actions = []
    for item in checks:
        if item["status"] in {"failed", "unideal", "unavailable"}:
            actions.append(
                {
                    "check": item["name"],
                    "priority": "high"
                    if item["hard_gate"] or item["status"] == "failed"
                    else "medium",
                    "action": item["remediation"],
                }
            )
    return actions[:10]


def higher_status(value, thresholds):
    if value is None:
        return "unavailable"
    if value < thresholds[0]:
        return "failed"
    if value < thresholds[1]:
        return "acceptable"
    if value < thresholds[2]:
        return "passed"
    return "strong"


def lower_status(value, thresholds):
    if value is None:
        return "unavailable"
    if value > thresholds[0]:
        return "failed"
    if value > thresholds[1]:
        return "unideal"
    if value > thresholds[2]:
        return "acceptable"
    return "strong"


def worst_status(statuses):
    priority = {
        "failed": 5,
        "unideal": 4,
        "acceptable": 3,
        "passed": 2,
        "strong": 1,
        "unavailable": 0,
    }
    values = list(statuses)
    if not values:
        return "unavailable"
    available = [status for status in values if status != "unavailable"]
    if not available:
        return "unavailable"
    return max(available, key=lambda status: priority.get(status, 6))


def load_csv(path):
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def load_json(path):
    if not path:
        return None
    json_path = Path(path)
    if not json_path.exists():
        return None
    return json.loads(json_path.read_text(encoding="utf-8"))


def artifact_hashes(paths):
    hashes = []
    for path in paths:
        file_path = Path(path)
        hashes.append(
            {
                "path": path,
                "exists": file_path.exists(),
                "sha256": sha256(file_path) if file_path.exists() else None,
            }
        )
    return hashes


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rounded(value, digits=3):
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value
