import re
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "Datasets",
    "datasets",
    "MedicalAgent",
    "Data",
    "KnowledgeBase/raw",
    "KnowledgeBase/processed",
    "node_modules",
    "dist",
    "playwright-report",
    "test-results",
}
SKIP_SUFFIXES = {".db", ".sqlite", ".nii", ".gz", ".dcm", ".png", ".jpg", ".jpeg", ".xlsx", ".zip", ".pt", ".joblib", ".pyc"}
ALLOWLIST = {
    "replace_with_your_groq_key",
    "replace_with_a_long_random_secret",
    "eval-only-signing-secret",
}

PATTERNS = [
    ("groq_api_key", re.compile(r"gsk_[A-Za-z0-9]{20,}")),
    ("openai_api_key", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    (
        "generic_secret_assignment",
        re.compile(
            r"(?i)(api[_-]?key|secret|token|password)[ \t]*=[ \t]*"
            r"['\"]?[A-Za-z0-9_\-]{24,}"
        ),
    ),
]


def main():
    findings = scan_secret_findings()
    if findings:
        for finding in findings:
            print(f"{finding['path']}:{finding['line']} possible {finding['pattern']}")
        raise SystemExit(1)
    print("Secret scan passed.")


def scan_secret_findings(root: Path = ROOT):
    findings = []
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        dirnames[:] = [name for name in dirnames if not _should_skip_dir(current / name, root)]
        for filename in filenames:
            path = current / filename
            if _should_skip(path, root):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for name, pattern in PATTERNS:
                for match in pattern.finditer(text):
                    value = match.group(0)
                    if any(allowed in value for allowed in ALLOWLIST):
                        continue
                    findings.append({
                        "pattern": name,
                        "path": str(path.relative_to(root)),
                        "line": text[:match.start()].count("\n") + 1,
                    })
    return findings


def _should_skip(path, root: Path = ROOT):
    relative = path.relative_to(root)
    if path.name.startswith(".env") and path.name != ".env.example":
        return True
    parts = set(relative.parts)
    if parts & SKIP_DIRS:
        return True
    as_posix = relative.as_posix()
    if any(as_posix.startswith(skip + "/") for skip in SKIP_DIRS):
        return True
    if path.suffix.lower() in SKIP_SUFFIXES:
        return True
    return False


def _should_skip_dir(path: Path, root: Path = ROOT) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    parts = set(relative.parts)
    if parts & SKIP_DIRS:
        return True
    as_posix = relative.as_posix()
    return any(as_posix == skip or as_posix.startswith(skip + "/") for skip in SKIP_DIRS)


if __name__ == "__main__":
    main()
