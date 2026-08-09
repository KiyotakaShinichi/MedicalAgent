"""Reproducible release evidence for the fail-closed RAG boundary."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from backend.services.rag_evidence_envelope import (
    EVIDENCE_ENVELOPE_VERSION,
    EVIDENCE_POLICY_VERSION,
    SAFETY_POLICY_VERSION,
    VALIDATOR_POLICY_VERSION,
    EvidenceDisposition,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = ROOT / "Data/evals/safety/latest_fail_closed_rag_assurance.json"
TEST_PATH = ROOT / "tests/test_rag_fail_closed_evidence_envelope.py"
MINIMUM_FAULT_CASES = 30
PROTECTED_RESPONSE_PATHS = (
    "patient_json_chat",
    "clinician_json_chat",
    "patient_sse_chat",
    "support_chat_persistence",
    "exact_cache_hit",
    "semantic_cache_hit",
    "live_agent_pipeline",
    "evaluation_pipeline",
)
SOURCE_PATHS = (
    "backend/services/rag_evidence_envelope.py",
    "backend/services/agent_post_gen.py",
    "backend/services/agent_rag.py",
    "backend/services/agent_cache.py",
    "backend/services/support_chat_agent.py",
    "backend/api/routers/patient_interactions.py",
    "tests/test_rag_fail_closed_evidence_envelope.py",
)


def _count_summary(output: str, label: str) -> int:
    matches = re.findall(rf"(?<!\w)(\d+)\s+{re.escape(label)}\b", output.lower())
    return max((int(value) for value in matches), default=0)


def _source_digests(root: Path, paths: Sequence[str] = SOURCE_PATHS) -> dict[str, str]:
    digests: dict[str, str] = {}
    for relative in paths:
        path = root / relative
        if path.exists():
            digests[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def build_fail_closed_assurance_report(
    *,
    return_code: int,
    output: str,
    duration_seconds: float,
    root: Path = ROOT,
    timed_out: bool = False,
) -> dict[str, Any]:
    passed = _count_summary(output, "passed")
    failed = _count_summary(output, "failed")
    errors = _count_summary(output, "error") + _count_summary(output, "errors")
    suite_satisfied = (
        return_code == 0
        and not timed_out
        and passed >= MINIMUM_FAULT_CASES
        and failed == 0
        and errors == 0
    )
    return {
        "schema_version": "fail_closed_rag_assurance_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if suite_satisfied else "failed",
        "invariant": "No valid evidence envelope -> no evidence-dependent medical answer.",
        "summary": {
            "passed_tests": passed,
            "failed_tests": failed,
            "error_tests": errors,
            "minimum_required_cases": MINIMUM_FAULT_CASES,
            "pytest_exit_code": return_code,
            "timed_out": timed_out,
            "duration_seconds": round(duration_seconds, 3),
        },
        "contract": {
            "envelope_version": EVIDENCE_ENVELOPE_VERSION,
            "release_policy_version": EVIDENCE_POLICY_VERSION,
            "safety_policy_version": SAFETY_POLICY_VERSION,
            "validator_policy_version": VALIDATOR_POLICY_VERSION,
            "closed_dispositions": [item.value for item in EvidenceDisposition],
            "only_allow_releases_evidence": True,
        },
        "coverage": {
            "protected_response_paths": list(PROTECTED_RESPONSE_PATHS),
            "all_known_response_paths_enforced": suite_satisfied,
            "cache_revalidation_required": True,
            "streaming_is_buffered_until_authorized": True,
            "taglish_abstention_parity_tested": True,
            "validator_fault_injection_tested": True,
        },
        "provenance": {
            "test_command": [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_rag_fail_closed_evidence_envelope.py",
                "-q",
            ],
            "source_sha256": _source_digests(root),
            "raw_patient_content_retained": False,
        },
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "claim_boundary": (
            "This is repeatable engineering evidence for availability and release integrity. "
            "It does not prove medical correctness, clinical validation, real-world safety, "
            "or production healthcare readiness."
        ),
    }


def run_fail_closed_assurance(
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    root: Path = ROOT,
    timeout_seconds: int = 180,
    executor: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_rag_fail_closed_evidence_envelope.py",
        "-q",
    ]
    started = time.perf_counter()
    timed_out = False
    try:
        completed = executor(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
        output = f"{completed.stdout or ''}\n{completed.stderr or ''}"
        return_code = int(completed.returncode)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        output = f"{exc.stdout or ''}\n{exc.stderr or ''}"
        return_code = 124
    report = build_fail_closed_assurance_report(
        return_code=return_code,
        output=output,
        duration_seconds=time.perf_counter() - started,
        root=root,
        timed_out=timed_out,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


__all__ = [
    "DEFAULT_OUTPUT_PATH",
    "MINIMUM_FAULT_CASES",
    "PROTECTED_RESPONSE_PATHS",
    "build_fail_closed_assurance_report",
    "run_fail_closed_assurance",
]
