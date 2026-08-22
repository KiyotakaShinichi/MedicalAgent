"""Validate NLCare's canonical dependency and compatibility manifests.

Source of truth
---------------
``pyproject.toml`` plus ``uv.lock`` are canonical. ``uv sync --frozen`` is the
only supported way to build a development or CI environment.

``requirements.txt`` and ``requirements-serving.txt`` are **not** duplicates and
are not obsolete: the container image installs with plain ``pip`` and never
runs ``uv`` (see ``Dockerfile``, ``ARG REQUIREMENTS_FILE=requirements.txt``),
and ``requirements-serving.txt`` is the deliberately smaller runtime profile
that omits training, deep-learning, SHAP, and evaluation dependencies.

Because they are hand-maintained exports of a canonical source, they can drift
from it silently — a version bumped in ``pyproject.toml`` but not in
``requirements.txt`` ships a *different* dependency set to the container than
the one every test ran against. Pinning alone never caught that: both files
stayed perfectly pinned to disagreeing versions.

This module therefore enforces three things:

1. every manifest is exact-pinned;
2. every runtime dependency in ``pyproject.toml`` appears in
   ``requirements.txt`` at the *same* version;
3. ``requirements-serving.txt`` is a subset of ``requirements.txt`` with
   matching versions.

Extra entries in ``requirements.txt`` are allowed — it doubles as the
pip-only development profile, so it legitimately carries a test runner — but an
extra entry that also appears in ``pyproject.toml`` must agree with it.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PINNED_REQUIREMENT = re.compile(
    r"^[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?==[^\s;]+(?:\s*;.*)?$"
)
# name, optional extras, pinned version.
_REQUIREMENT_PARTS = re.compile(
    r"^(?P<name>[A-Za-z0-9_.-]+)(?:\[[A-Za-z0-9_,.-]+\])?==(?P<version>[^\s;]+)"
)


def _canonical_name(name: str) -> str:
    """PEP 503 normalisation, so `types-PyYAML` and `types_pyyaml` compare equal."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _pinned_versions(requirements: list[str]) -> dict[str, str]:
    """Map canonical distribution name -> pinned version, ignoring markers."""
    versions: dict[str, str] = {}
    for requirement in requirements:
        match = _REQUIREMENT_PARTS.match(requirement)
        if match:
            versions[_canonical_name(match.group("name"))] = match.group("version")
    return versions


def _active_requirements(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def validate_dependency_contract(root: Path = ROOT) -> list[str]:
    issues: list[str] = []
    pyproject_path = root / "pyproject.toml"
    lock_path = root / "uv.lock"
    if not pyproject_path.is_file():
        return ["missing pyproject.toml"]
    if not lock_path.is_file():
        issues.append("missing uv.lock")

    project = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    dependencies = project.get("project", {}).get("dependencies", [])
    if not dependencies:
        issues.append("project.dependencies is empty")
    for requirement in dependencies:
        if "==" not in str(requirement):
            issues.append(f"unpinned project dependency: {requirement}")

    manifests: dict[str, dict[str, str]] = {}
    for relative_path in ("requirements.txt", "requirements-serving.txt"):
        path = root / relative_path
        if not path.is_file():
            issues.append(f"missing {relative_path}")
            continue
        active = _active_requirements(path)
        for requirement in active:
            if not PINNED_REQUIREMENT.match(requirement):
                issues.append(f"unpinned compatibility dependency in {relative_path}: {requirement}")
        manifests[relative_path] = _pinned_versions(active)

    issues.extend(_drift_issues(project, dependencies, manifests))
    return issues


def _drift_issues(
    project: dict,
    dependencies: list,
    manifests: dict[str, dict[str, str]],
) -> list[str]:
    """Compatibility manifests must agree with the canonical source.

    Pinning is not agreement: two files can both be exact-pinned and still
    disagree, which is the failure that ships one dependency set to the
    container and a different one to every test.
    """
    issues: list[str] = []
    canonical = _pinned_versions([str(d) for d in dependencies])
    full = manifests.get("requirements.txt")
    serving = manifests.get("requirements-serving.txt")

    if full is not None:
        for name, version in sorted(canonical.items()):
            if name not in full:
                issues.append(
                    f"requirements.txt is missing runtime dependency {name}=={version} "
                    "declared in pyproject.toml"
                )
            elif full[name] != version:
                issues.append(
                    f"dependency drift for {name}: pyproject.toml pins {version} "
                    f"but requirements.txt pins {full[name]}"
                )

        # `requirements.txt` doubles as the pip-only development profile, so
        # extras are fine — but an extra that pyproject also declares (in a
        # dependency group) must not disagree with it.
        grouped: dict[str, str] = {}
        for group in (project.get("dependency-groups") or {}).values():
            grouped.update(_pinned_versions([str(entry) for entry in group]))
        for name, version in sorted(grouped.items()):
            if name in full and full[name] != version:
                issues.append(
                    f"dependency drift for {name}: pyproject.toml dependency group pins "
                    f"{version} but requirements.txt pins {full[name]}"
                )

    if full is not None and serving is not None:
        for name, version in sorted(serving.items()):
            if name not in full:
                issues.append(
                    f"requirements-serving.txt declares {name}=={version}, which is absent "
                    "from requirements.txt; the serving profile must be a subset"
                )
            elif full[name] != version:
                issues.append(
                    f"dependency drift for {name}: requirements.txt pins {full[name]} "
                    f"but requirements-serving.txt pins {version}"
                )
    return issues


def main() -> int:
    issues = validate_dependency_contract()
    if issues:
        print("Dependency contract: FAILED")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print("Dependency contract: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

