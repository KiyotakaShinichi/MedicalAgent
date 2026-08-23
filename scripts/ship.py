"""Cross-platform NLCare ship gate.

Runs the same checks as ``make ship`` without requiring GNU Make. The script
stops on the first failed command and returns that command's exit code.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# `python scripts/ship.py` puts only `scripts/` on sys.path, so the repository
# root is added before importing the step package. ROOT itself then comes from
# that package, which is the single definition both sides share.
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ship_steps import FRONTEND, ROOT, Step, all_steps  # noqa: E402

SHIP_MANIFEST = ROOT / "Data" / "evals" / "ops" / "latest_ship_run.json"
DEFAULT_STEP_TIMEOUT_SECONDS = 900
FAST_MANIFEST = ROOT / "Data" / "evals" / "ops" / "latest_ship_fast_run.json"
EVIDENCE_MANIFEST = (
    ROOT / "Data" / "evals" / "ops" / "latest_ship_evidence_run.json"
)
FAST_STEP_NAMES = {
    "Backend breast-monitoring integration tests",
    "Fail-closed RAG release assurance",
    "Restricted synthetic staging assurance",
    "Backend progressive-loading and notification reliability tests",
    "Cloud, data-platform, and managed-vector contract tests",
    "Assurance, XAI, automation, and safety contract tests",
    "SaaS control-plane and worker contract tests",
    "Frontend Vitest unit tests",
    "Frontend lint",
    "Frontend production build",
}
_FILE_DIGEST_CACHE: dict[tuple[str, int, int], bytes] = {}


def _effective_timeout(step: Step) -> int:
    if step.timeout_seconds is not None:
        return max(30, int(step.timeout_seconds))
    configured = os.getenv("NLCARE_SHIP_STEP_TIMEOUT_SECONDS")
    if configured:
        try:
            return max(30, int(configured))
        except ValueError:
            pass
    return DEFAULT_STEP_TIMEOUT_SECONDS


def _run(
    step: Step, *, dependency_fingerprint: str | None = None
) -> dict[str, object]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if step.env:
        env.update(step.env)
    print(f"\n[ship] {step.name}", flush=True)
    print(f"[ship] cwd={step.cwd}", flush=True)
    print(f"[ship] cmd={' '.join(step.command)}", flush=True)
    timeout_seconds = _effective_timeout(step)
    started = time.perf_counter()
    subprocess.run(
        step.command,
        cwd=step.cwd,
        env=env,
        check=True,
        timeout=timeout_seconds,
    )
    elapsed = round(time.perf_counter() - started, 3)
    print(f"[ship] passed in {elapsed}s", flush=True)
    return {
        "name": step.name,
        "status": "passed",
        "duration_seconds": elapsed,
        "timeout_seconds": timeout_seconds,
        "cwd": str(step.cwd.relative_to(ROOT) if step.cwd != ROOT else "."),
        "command": step.command,
        "dependency_fingerprint": dependency_fingerprint,
    }


def _write_manifest(
    *,
    status: str,
    step_results: list[dict[str, object]],
    failed_step: str | None = None,
    failure_kind: str | None = None,
    tier: str = "release",
    resume_requested: bool = False,
    selected_step_count: int | None = None,
    output_path: Path | None = None,
) -> None:
    target = output_path or SHIP_MANIFEST
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "nlcare_ship_run_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "tier": tier,
        "resume_requested": resume_requested,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "completed_step_count": len(step_results),
        "selected_step_count": selected_step_count or len(step_results),
        "cached_step_count": sum(
            result.get("status") == "cached_pass" for result in step_results
        ),
        "failed_step": failed_step,
        "failure_kind": failure_kind,
        "steps": step_results,
        "claim_boundary": (
            "This manifest records local engineering gate execution only. A passing ship run "
            "does not establish clinical validation, real-world safety, compliance, or "
            "production healthcare readiness."
        ),
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_steps() -> list[Step]:
    """Every ship step, in execution order.

    The definitions live in ``scripts/ship_steps``, grouped by responsibility.
    Order across those groups is the contract this run depends on - later steps
    consume evidence earlier ones produce - so it is fixed by ``STEP_GROUPS``
    rather than by import order.
    """
    return all_steps()


def _build_post_success_reconciliation_steps() -> list[Step]:
    return [
        Step(
            name="Post-ship evidence reconciliation",
            command=[
                sys.executable,
                "scripts/run_post_ship_evidence_reconciliation.py",
            ],
            timeout_seconds=300,
        ),
    ]


def _manifest_path_for_tier(tier: str) -> Path:
    if tier == "fast":
        return FAST_MANIFEST
    if tier == "evidence":
        return EVIDENCE_MANIFEST
    return SHIP_MANIFEST


def _is_evidence_step(step: Step) -> bool:
    return any(
        part.replace("\\", "/").startswith("scripts/")
        and part.endswith(".py")
        for part in step.command
    )


def _select_steps(steps: list[Step], tier: str) -> list[Step]:
    if tier == "release":
        return steps
    if tier == "fast":
        return [step for step in steps if step.name in FAST_STEP_NAMES]
    if tier == "evidence":
        return [step for step in steps if _is_evidence_step(step)]
    raise ValueError(f"unsupported ship tier: {tier}")


def _candidate_dependency_paths(step: Step) -> list[Path]:
    paths: list[Path] = [Path(__file__).resolve()]
    if step.cwd == FRONTEND:
        paths.extend(
            [
                FRONTEND / "src",
                FRONTEND / "tests",
                FRONTEND / "package.json",
                FRONTEND / "package-lock.json",
                FRONTEND / "vite.config.ts",
                FRONTEND / "vitest.config.ts",
                FRONTEND / "playwright.config.ts",
                FRONTEND / "tsconfig.json",
            ]
        )
        return paths

    paths.extend([ROOT / "backend", ROOT / "config"])
    for part in step.command:
        normalized = part.replace("\\", "/")
        if normalized.endswith(".py"):
            candidate = ROOT / normalized
            if candidate.exists():
                paths.append(candidate)
    if "-m" in step.command and "pytest" in step.command:
        paths.append(ROOT / "tests" / "conftest.py")
    return paths


def _iter_dependency_files(paths: list[Path]):
    seen: set[Path] = set()
    excluded = {
        ".git",
        "__pycache__",
        "node_modules",
        "dist",
        "build",
        ".pytest_cache",
    }
    for path in paths:
        if path.is_file():
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield resolved
            continue
        if not path.exists():
            continue
        for candidate in sorted(path.rglob("*")):
            if not candidate.is_file():
                continue
            if excluded.intersection(candidate.parts):
                continue
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield resolved


def _file_digest(path: Path) -> bytes:
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _FILE_DIGEST_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256(path.read_bytes()).digest()
    _FILE_DIGEST_CACHE[key] = digest
    return digest


def _dependency_fingerprint(step: Step) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(step.command, sort_keys=True).encode("utf-8"))
    digest.update(str(step.cwd.resolve()).encode("utf-8"))
    digest.update(json.dumps(step.env or {}, sort_keys=True).encode("utf-8"))
    for path in _iter_dependency_files(_candidate_dependency_paths(step)):
        try:
            relative = path.relative_to(ROOT)
        except ValueError:
            relative = path
        digest.update(str(relative).replace("\\", "/").encode("utf-8"))
        digest.update(_file_digest(path))
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _cached_result(
    previous: dict[str, object] | None,
    step: Step,
    fingerprint: str,
) -> dict[str, object] | None:
    if not previous:
        return None
    rows = previous.get("steps")
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, dict) or row.get("name") != step.name:
            continue
        if row.get("status") not in {"passed", "cached_pass"}:
            return None
        if row.get("dependency_fingerprint") != fingerprint:
            return None
        return {
            "name": step.name,
            "status": "cached_pass",
            "duration_seconds": 0.0,
            "timeout_seconds": _effective_timeout(step),
            "cwd": str(
                step.cwd.relative_to(ROOT) if step.cwd != ROOT else "."
            ),
            "command": step.command,
            "dependency_fingerprint": fingerprint,
            "reused_from": previous.get("generated_at"),
        }
    return None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier",
        choices=("fast", "evidence", "release"),
        default="release",
        help="fast=core tests/build, evidence=artifact refresh, release=all gates",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only prior passed steps with identical dependency fingerprints.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List selected steps without running them.",
    )
    return parser.parse_args(argv)


def _failure_row(
    step: Step, *, fingerprint: str, exit_code: int | None
) -> dict[str, object]:
    """Manifest row for a step that did not pass.

    `exit_code=None` means the step was killed by its timeout, and
    `duration_seconds` is then the timeout itself: the step ran that long before
    being stopped.
    """
    timeout_seconds = _effective_timeout(step)
    timed_out = exit_code is None
    row: dict[str, object] = {
        "name": step.name,
        "status": "timed_out" if timed_out else "failed",
        "duration_seconds": timeout_seconds if timed_out else None,
        "timeout_seconds": timeout_seconds,
        "cwd": str(step.cwd.relative_to(ROOT) if step.cwd != ROOT else "."),
        "command": step.command,
        "dependency_fingerprint": fingerprint,
    }
    if not timed_out:
        row["exit_code"] = exit_code
    return row


def _execute_steps(
    steps: list[Step],
    *,
    args: argparse.Namespace,
    step_results: list[dict[str, object]],
    manifest_path: Path,
    previous: dict | None,
    selected_step_count: int,
    failure_kind_prefix: str = "",
) -> int | None:
    """Run `steps`, appending a manifest row for each.

    Returns the process exit code on the first failure, or None if every step
    passed. The main pass and the post-success reconciliation pass differ only
    in whether resume caching applies, how failures are labelled, and the step
    count recorded in the manifest, so they share this body rather than
    duplicating the failure handling twice over.
    """
    for step in steps:
        fingerprint = _dependency_fingerprint(step)
        cached = (
            _cached_result(previous, step, fingerprint)
            if previous is not None
            else None
        )
        if cached is not None:
            print(f"\n[ship] cached: {step.name}", flush=True)
            step_results.append(cached)
            continue
        try:
            step_results.append(_run(step, dependency_fingerprint=fingerprint))
            continue
        except subprocess.TimeoutExpired:
            exit_code: int | None = None
            failure_kind = f"{failure_kind_prefix}timeout"
            returncode = 124
            detail = f"timed out after {_effective_timeout(step)}s"
        except subprocess.CalledProcessError as exc:
            exit_code = int(exc.returncode or 1)
            failure_kind = f"{failure_kind_prefix}nonzero_exit"
            returncode = exit_code
            detail = f"exited {exc.returncode}"

        step_results.append(
            _failure_row(step, fingerprint=fingerprint, exit_code=exit_code)
        )
        _write_manifest(
            status="failed",
            step_results=step_results,
            failed_step=step.name,
            failure_kind=failure_kind,
            tier=args.tier,
            resume_requested=args.resume,
            selected_step_count=selected_step_count,
            output_path=manifest_path,
        )
        print(f"\n[ship] FAILED: {step.name} {detail}", file=sys.stderr, flush=True)
        return returncode
    return None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    steps = _select_steps(_build_steps(), args.tier)
    manifest_path = _manifest_path_for_tier(args.tier)
    if args.list:
        for index, step in enumerate(steps, start=1):
            print(f"{index:02d}. {step.name}")
        return 0

    previous = _load_manifest(manifest_path) if args.resume else None
    step_results: list[dict[str, object]] = []

    failed = _execute_steps(
        steps,
        args=args,
        step_results=step_results,
        manifest_path=manifest_path,
        previous=previous,
        selected_step_count=len(steps),
    )
    if failed is not None:
        return failed

    _write_manifest(
        status="passed",
        step_results=step_results,
        tier=args.tier,
        resume_requested=args.resume,
        selected_step_count=len(steps),
        output_path=manifest_path,
    )

    # Reconciliation refreshes the evidence the just-passed steps invalidated,
    # so it runs only after a green release tier, and is never resume-cached:
    # its inputs are exactly the artifacts this run just rewrote.
    reconciliation_steps = (
        _build_post_success_reconciliation_steps() if args.tier == "release" else []
    )
    if reconciliation_steps:
        total_step_count = len(steps) + len(reconciliation_steps)
        failed = _execute_steps(
            reconciliation_steps,
            args=args,
            step_results=step_results,
            manifest_path=manifest_path,
            previous=None,
            selected_step_count=total_step_count,
            failure_kind_prefix="post_success_reconciliation_",
        )
        if failed is not None:
            return failed

        _write_manifest(
            status="passed",
            step_results=step_results,
            tier=args.tier,
            resume_requested=args.resume,
            selected_step_count=total_step_count,
            output_path=manifest_path,
        )
    print(
        f"\n[ship] PASSED: {args.tier} tier green "
        f"({sum(row['status'] == 'cached_pass' for row in step_results)} cached)",
        flush=True,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
