"""Fail when source-referenced application settings are absent from .env.example."""

from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (ROOT / "backend", ROOT / "scripts")
PLATFORM_MANAGED = {"LOCALAPPDATA", "PYTEST_CURRENT_TEST", "PYTHON"}


def source_environment_names() -> set[str]:
    names: set[str] = set()
    for source_root in SOURCE_ROOTS:
        for path in source_root.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                name = _environment_name(node)
                if name:
                    names.add(name)
    return names - PLATFORM_MANAGED


def _environment_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call) and node.args:
        first = node.args[0]
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            return None
        function = node.func
        if (
            isinstance(function, ast.Attribute)
            and function.attr == "getenv"
            and isinstance(function.value, ast.Name)
            and function.value.id == "os"
        ):
            return first.value
        if (
            isinstance(function, ast.Attribute)
            and function.attr in {"get", "setdefault"}
            and isinstance(function.value, ast.Attribute)
            and isinstance(function.value.value, ast.Name)
            and function.value.value.id == "os"
            and function.value.attr == "environ"
        ):
            return first.value
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "os"
        and node.value.attr == "environ"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    ):
        return node.slice.value
    return None


def documented_environment_names() -> set[str]:
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    return set(re.findall(r"^([A-Z][A-Z0-9_]+)=", text, flags=re.MULTILINE))


def undocumented_environment_names() -> list[str]:
    return sorted(source_environment_names() - documented_environment_names())


def main() -> int:
    missing = undocumented_environment_names()
    if missing:
        print("Undocumented source-referenced environment variables:")
        print("\n".join(missing))
        return 1
    print(f"Environment contract documented: {len(source_environment_names())} variables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
