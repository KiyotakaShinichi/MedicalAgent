from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_demo_stack.ps1"


def _launcher_text() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


def test_demo_launcher_bootstraps_both_scoped_patients() -> None:
    text = _launcher_text()

    assert "scripts.bootstrap_synthetic_demo" in text
    assert "P001 / patient-demo" in text
    assert "P002 / patient-demo" in text


def test_demo_launcher_rejects_occupied_ports_before_starting() -> None:
    text = _launcher_text()

    assert "Assert-PortAvailable -Port $ApiPort" in text
    assert "Assert-PortAvailable -Port $FrontendPort" in text
    assert "Get-NetTCPConnection -LocalPort $Port -State Listen" in text


def test_demo_launcher_waits_for_backend_and_frontend_readiness() -> None:
    text = _launcher_text()

    assert 'Wait-ForHttpReady -Url "http://127.0.0.1:$ApiPort/health"' in text
    assert 'Wait-ForHttpReady -Url "http://127.0.0.1:$FrontendPort/login"' in text
    assert "Both services passed their HTTP readiness checks." in text


def test_demo_launcher_uses_stable_backend_worker_without_reload() -> None:
    text = _launcher_text()

    assert "uvicorn backend.api.main:app" in text
    assert "--reload" not in text


def test_demo_launcher_defers_heavy_patient_enrichment_until_requested() -> None:
    text = _launcher_text()

    assert "NLCARE_PATIENT_ENRICHMENT_PREWARM_ENABLED='false'" in text
