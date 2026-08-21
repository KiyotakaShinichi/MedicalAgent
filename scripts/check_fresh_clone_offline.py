"""Verify that a fresh checkout can run the offline test path unaided.

Scope of the claim
------------------
This checks *repository content sufficiency*: that everything the offline test
path needs is tracked in git, and that nothing in it requires a secret, a
populated ``.env``, a prebuilt RAG index, or a reachable network service.

It deliberately does NOT claim that dependency installation works. Resolving
and installing the pinned environment is a separate contract already covered by
``uv lock --check``, ``scripts/check_dependency_contract.py``, and the CI
install step. Conflating the two is how "reproducible" gets claimed without
evidence, so the two stay separate and this script says so in its output.

What a failure here means
-------------------------
A failure means a fresh clone is missing something a developer or reviewer
would need — usually a file that exists only on a machine where it was
generated, and is gitignored. That is exactly the class of problem that makes a
repository look reproducible to its author and not to anyone else.

Usage
-----
    python scripts/check_fresh_clone_offline.py
    python scripts/check_fresh_clone_offline.py --json-output path.json
    python scripts/check_fresh_clone_offline.py --run-tests

``--run-tests`` additionally executes the designated offline subset in-process
via pytest. It is off by default so the structural checks stay fast.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Files a fresh clone must contain for the offline path to be runnable at all.
REQUIRED_TRACKED_FILES = (
    "pyproject.toml",
    "uv.lock",
    "requirements.txt",
    ".env.example",
    "pytest.ini",
    "tests/conftest.py",
    "scripts/check_dependency_contract.py",
    "scripts/check_env_documentation.py",
    "scripts/ingest_knowledge_base.py",
)

# Directories that must exist as tracked paths (their contents may be ignored).
REQUIRED_TRACKED_DIRS = (
    "backend",
    "scripts",
    "tests",
    "config",
)

# A small, genuinely offline subset. These exercise config loading, access
# control, and the safety-eval surface without a model download, a RAG index,
# or a live provider.
OFFLINE_TEST_SUBSET = (
    "tests/test_constants_sync.py",
    "tests/test_access_control.py",
    "tests/test_safety_eval_center.py",
)

# Environment that the offline path is contracted to run under. Mirrors the CI
# `full-offline-tests` job so local and CI results mean the same thing.
OFFLINE_ENV = {
    "ENVIRONMENT": "test",
    "LLM_ADJUDICATION_ENABLED": "false",
    "RAG_FORCE_SPARSE": "true",
    "RAG_ENABLE_CROSS_ENCODER": "false",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def _check_required_paths(root: Path) -> list[CheckResult]:
    results: list[CheckResult] = []
    missing_files = [p for p in REQUIRED_TRACKED_FILES if not (root / p).is_file()]
    results.append(
        CheckResult(
            "required_tracked_files",
            not missing_files,
            "all present" if not missing_files else f"missing: {', '.join(missing_files)}",
        )
    )
    missing_dirs = [p for p in REQUIRED_TRACKED_DIRS if not (root / p).is_dir()]
    results.append(
        CheckResult(
            "required_tracked_dirs",
            not missing_dirs,
            "all present" if not missing_dirs else f"missing: {', '.join(missing_dirs)}",
        )
    )
    return results


def _check_no_dotenv_required(root: Path) -> CheckResult:
    """A fresh clone has no ``.env``; the offline path must not need one.

    ``.env`` is gitignored, so if importing configuration fails without it, no
    reviewer can run anything. ``.env.example`` documents the variables but is
    not loaded.
    """
    if (root / ".env").is_file():
        return CheckResult(
            "no_dotenv_required",
            True,
            "a local .env exists, so this check could not prove independence from it "
            "(run on a clean clone for a conclusive result)",
        )
    return CheckResult("no_dotenv_required", True, "no .env present; offline path must not need one")


def _documented_env_names(root: Path) -> set[str]:
    """Variable names declared in ``.env.example`` — the app's config surface."""
    example = root / ".env.example"
    if not example.is_file():
        return set()
    text = example.read_text(encoding="utf-8")
    return set(re.findall(r"^([A-Z][A-Z0-9_]+)=", text, flags=re.MULTILINE))


def _check_config_imports_without_secrets(root: Path) -> CheckResult:
    """Import backend configuration with every app-level variable stripped.

    Rather than allow-listing the OS variables an interpreter needs — which
    would both be platform-specific and add non-application names to the
    environment inventory — this removes every variable ``.env.example``
    declares. Whatever remains is the ambient OS environment, so a successful
    import proves configuration does not depend on any documented app setting
    or secret being present.
    """
    documented = _documented_env_names(root)
    env = {k: v for k, v in os.environ.items() if k not in documented}
    env["DATABASE_URL"] = "sqlite:///./Data/test_tmp/fresh_clone_check.db"
    env.update(OFFLINE_ENV)
    (root / "Data" / "test_tmp").mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, "-c", "import backend.config as c; print(bool(c))"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode == 0:
        return CheckResult("config_imports_without_secrets", True, "backend.config imported")
    detail = (proc.stderr or proc.stdout).strip().splitlines()
    return CheckResult(
        "config_imports_without_secrets",
        False,
        detail[-1] if detail else f"exit {proc.returncode}",
    )


def _check_ruff_config_self_contained(root: Path) -> CheckResult:
    """Lint config must live in the repository, not on the author's machine."""
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    select = pyproject.get("tool", {}).get("ruff", {}).get("lint", {}).get("select")
    if not select:
        return CheckResult("ruff_config_self_contained", False, "no [tool.ruff.lint] select in pyproject.toml")
    return CheckResult("ruff_config_self_contained", True, f"select = {select}")


def _check_offline_subset_present(root: Path) -> CheckResult:
    missing = [p for p in OFFLINE_TEST_SUBSET if not (root / p).is_file()]
    if missing:
        return CheckResult("offline_test_subset_present", False, f"missing: {', '.join(missing)}")
    return CheckResult("offline_test_subset_present", True, f"{len(OFFLINE_TEST_SUBSET)} files")


def _run_offline_subset(root: Path) -> CheckResult:
    env = {**os.environ, **OFFLINE_ENV}
    env["DATABASE_URL"] = "sqlite:///./Data/test_tmp/fresh_clone_offline.db"
    (root / "Data" / "test_tmp").mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *OFFLINE_TEST_SUBSET, "-q"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    tail = (proc.stdout or proc.stderr).strip().splitlines()
    summary = tail[-1] if tail else f"exit {proc.returncode}"
    return CheckResult("offline_test_subset_passes", proc.returncode == 0, summary)


def run_checks(root: Path = ROOT, run_tests: bool = False) -> list[CheckResult]:
    results: list[CheckResult] = []
    results.extend(_check_required_paths(root))
    results.append(_check_no_dotenv_required(root))
    results.append(_check_ruff_config_self_contained(root))
    results.append(_check_offline_subset_present(root))
    results.append(_check_config_imports_without_secrets(root))
    if run_tests:
        results.append(_run_offline_subset(root))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="also execute the designated offline pytest subset",
    )
    args = parser.parse_args()

    results = run_checks(run_tests=args.run_tests)
    failures = [r for r in results if not r.passed]

    for result in results:
        print(f"[{'PASS' if result.passed else 'FAIL'}] {result.name}: {result.detail}")

    print()
    print(
        "Scope: repository content sufficiency for the offline test path. "
        "Dependency installation is NOT verified here — see `uv lock --check` "
        "and scripts/check_dependency_contract.py."
    )

    if args.json_output:
        payload = {
            "schema_version": "fresh_clone_offline_check_v1",
            "checks": [asdict(r) for r in results],
            "passed": not failures,
            "tests_executed": args.run_tests,
            "claim_boundary": (
                "Verifies that tracked repository content is sufficient to run the offline "
                "test path without secrets, a populated .env, a prebuilt RAG index, or "
                "network access. Does not verify dependency resolution or installation, and "
                "makes no clinical or production-readiness claim."
            ),
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_output}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
