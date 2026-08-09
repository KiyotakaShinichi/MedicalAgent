from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_final_backend_image_is_distroless_and_nonroot() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM gcr.io/distroless/python3-debian13:nonroot" in dockerfile
    assert "USER nonroot:nonroot" in dockerfile
    assert 'ENTRYPOINT ["/usr/bin/python3"]' in dockerfile
    runtime_stage = dockerfile.split("FROM gcr.io/distroless", 1)[1]
    assert "apt-get" not in runtime_stage
    assert "/bin/sh" not in runtime_stage


def test_staging_worker_uses_python_entrypoint_without_nested_python() -> None:
    compose = (ROOT / "docker-compose.synthetic-staging.yml").read_text(encoding="utf-8")
    assert 'command: ["scripts/run_automation_worker.py"' in compose
    assert 'command: ["python", "scripts/run_automation_worker.py"' not in compose
    assert "/usr/bin/python3" in compose
    assert "PYTHONPATH: /app" not in compose


def test_container_entrypoint_avoids_shell_execution() -> None:
    entrypoint = (ROOT / "scripts/container_entrypoint.py").read_text(encoding="utf-8")
    assert "os.execv" in entrypoint
    assert "shell=True" not in entrypoint
    assert "subprocess.run" in entrypoint
