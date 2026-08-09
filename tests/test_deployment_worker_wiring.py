from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_compose_profiles_launch_the_database_leased_worker():
    for name in ("docker-compose.yml", "docker-compose.prod.yml"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "scripts/run_automation_worker.py" in text
        assert "scripts/run_task_worker.py" not in text
        assert "--lease-seconds" in text


def test_durable_worker_runner_initializes_schema_before_polling():
    text = (ROOT / "scripts/run_automation_worker.py").read_text(encoding="utf-8")
    assert "ensure_schema()" in text
