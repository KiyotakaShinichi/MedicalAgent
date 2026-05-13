from sqlalchemy import inspect, text

from backend.database import Base, engine


def ensure_schema():
    import backend.models  # noqa: F401

    Base.metadata.create_all(bind=engine)

    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    lab_columns = {column["name"] for column in inspector.get_columns("lab_results")}
    cache_columns = set()
    if "agent_response_cache" in table_names:
        cache_columns = {column["name"] for column in inspector.get_columns("agent_response_cache")}

    with engine.begin() as connection:
        if "source" not in lab_columns:
            connection.execute(text("ALTER TABLE lab_results ADD COLUMN source VARCHAR DEFAULT 'manual' NOT NULL"))
        if "source_note" not in lab_columns:
            connection.execute(text("ALTER TABLE lab_results ADD COLUMN source_note TEXT"))
        if "knowledge_fingerprint" not in cache_columns:
            connection.execute(text("ALTER TABLE agent_response_cache ADD COLUMN knowledge_fingerprint VARCHAR"))
        if "cache_schema_version" not in cache_columns:
            connection.execute(text("ALTER TABLE agent_response_cache ADD COLUMN cache_schema_version VARCHAR"))
        if "cache_policy_json" not in cache_columns:
            connection.execute(text("ALTER TABLE agent_response_cache ADD COLUMN cache_policy_json TEXT"))
        if "expires_at" not in cache_columns:
            connection.execute(text("ALTER TABLE agent_response_cache ADD COLUMN expires_at DATETIME"))
        if "last_hit_at" not in cache_columns:
            connection.execute(text("ALTER TABLE agent_response_cache ADD COLUMN last_hit_at DATETIME"))

    rag_log_columns = set()
    if "rag_evaluation_logs" in table_names:
        rag_log_columns = {column["name"] for column in inspector.get_columns("rag_evaluation_logs")}

    with engine.begin() as connection:
        if "query_preview" not in rag_log_columns:
            connection.execute(text("ALTER TABLE rag_evaluation_logs ADD COLUMN query_preview VARCHAR"))
