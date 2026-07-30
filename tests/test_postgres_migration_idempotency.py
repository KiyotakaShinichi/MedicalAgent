from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_upgrade_paths_do_not_swallow_transactional_ddl_errors():
    migration = (
        ROOT
        / "backend"
        / "migrations"
        / "versions"
        / "0003_prediction_trace_and_rag_phase11.py"
    ).read_text(encoding="utf-8")
    upgrade = migration.split("def upgrade()", 1)[1].split("def downgrade()", 1)[0]
    assert "_existing_indexes(RAG_TABLE)" in upgrade
    assert "except Exception" not in upgrade
    assert 'bind.dialect.name == "postgresql"' in upgrade
    assert 'type_=sa.String(length=128)' in upgrade
