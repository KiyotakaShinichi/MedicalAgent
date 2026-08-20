"""Validate NLCare's canonical dependency and compatibility manifests."""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PINNED_REQUIREMENT = re.compile(
    r"^[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?==[^\s;]+(?:\s*;.*)?$"
)


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

    for relative_path in ("requirements.txt", "requirements-serving.txt"):
        path = root / relative_path
        if not path.is_file():
            issues.append(f"missing {relative_path}")
            continue
        for requirement in _active_requirements(path):
            if not PINNED_REQUIREMENT.match(requirement):
                issues.append(f"unpinned compatibility dependency in {relative_path}: {requirement}")
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

