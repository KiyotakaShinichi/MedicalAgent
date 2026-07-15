from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAYWRIGHT_CONFIG = ROOT / "frontend-react" / "playwright.config.ts"
BACKEND_RUNNER = ROOT / "scripts" / "run_playwright_backend.py"


def test_playwright_uses_ports_separate_from_live_demo() -> None:
    config = PLAYWRIGHT_CONFIG.read_text(encoding="utf-8")
    assert 'baseURL: "http://127.0.0.1:5273"' in config
    assert 'url: "http://127.0.0.1:8117/health"' in config
    assert 'VITE_API_BASE: "http://127.0.0.1:8117"' in config


def test_playwright_never_reuses_live_servers() -> None:
    config = PLAYWRIGHT_CONFIG.read_text(encoding="utf-8")
    assert config.count("reuseExistingServer: false") == 2
    assert "reuseExistingServer: true" not in config


def test_playwright_backend_uses_disposable_runtime_database() -> None:
    runner = BACKEND_RUNNER.read_text(encoding="utf-8")
    assert "playwright_test.db" in runner
    assert "reset_local_db.py" in runner
    assert '"DATABASE_URL": DATABASE_URL' in runner


def test_mutating_smoke_fixture_cannot_target_live_api() -> None:
    config = PLAYWRIGHT_CONFIG.read_text(encoding="utf-8")
    assert "http://127.0.0.1:8017" not in config
    assert "http://127.0.0.1:5173" not in config
