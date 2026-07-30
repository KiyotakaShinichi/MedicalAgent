from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _requirements(path: Path) -> set[str]:
    return {
        line.strip().lower().split("[", 1)[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def test_serving_profile_keeps_api_dependencies_and_excludes_offline_research():
    serving = _requirements(ROOT / "requirements-serving.txt")
    assert {"fastapi", "uvicorn", "sqlalchemy", "pandas", "scikit-learn"} <= serving
    assert {
        "torch",
        "torchvision",
        "shap",
        "matplotlib",
        "pytest",
        "sentence-transformers",
    }.isdisjoint(serving)


def test_full_research_profile_remains_available():
    full = _requirements(ROOT / "requirements.txt")
    assert {"torch", "torchvision", "shap", "sentence-transformers"} <= full
