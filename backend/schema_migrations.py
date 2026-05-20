from sqlalchemy import inspect, text

from backend.database import Base, engine


def ensure_schema():
    import backend.models  # noqa: F401

    Base.metadata.create_all(bind=engine)

    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    lab_columns = {column["name"] for column in inspector.get_columns("lab_results")}
    cache_columns = set()
    app_event_columns = set()
    if "agent_response_cache" in table_names:
        cache_columns = {column["name"] for column in inspector.get_columns("agent_response_cache")}
    if "app_event_logs" in table_names:
        app_event_columns = {column["name"] for column in inspector.get_columns("app_event_logs")}

    with engine.begin() as connection:
        if "source" not in lab_columns:
            connection.execute(text("ALTER TABLE lab_results ADD COLUMN source VARCHAR DEFAULT 'manual' NOT NULL"))
        if "source_note" not in lab_columns:
            connection.execute(text("ALTER TABLE lab_results ADD COLUMN source_note TEXT"))
        if "request_id" not in app_event_columns:
            connection.execute(text("ALTER TABLE app_event_logs ADD COLUMN request_id VARCHAR"))
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
        if "request_id" not in rag_log_columns:
            connection.execute(text("ALTER TABLE rag_evaluation_logs ADD COLUMN request_id VARCHAR"))
        # Canonical schema history lives in Alembic revision 0003.  These
        # idempotent startup patches keep existing local SQLite demo DBs from
        # silently returning null Phase 11 trace replay fields before the user
        # runs `alembic upgrade head`.
        for name, ddl in {
            "rag_mode": "ALTER TABLE rag_evaluation_logs ADD COLUMN rag_mode VARCHAR",
            "rewritten_query": "ALTER TABLE rag_evaluation_logs ADD COLUMN rewritten_query TEXT",
            "evidence_grade_json": "ALTER TABLE rag_evaluation_logs ADD COLUMN evidence_grade_json TEXT",
            "claim_validation_json": "ALTER TABLE rag_evaluation_logs ADD COLUMN claim_validation_json TEXT",
            "tier_filter_json": "ALTER TABLE rag_evaluation_logs ADD COLUMN tier_filter_json TEXT",
            "post_gen_validator_json": "ALTER TABLE rag_evaluation_logs ADD COLUMN post_gen_validator_json TEXT",
            "compound_intent_json": "ALTER TABLE rag_evaluation_logs ADD COLUMN compound_intent_json TEXT",
        }.items():
            if name not in rag_log_columns:
                connection.execute(text(ddl))

    family_columns = set()
    if "family_cancer_history_records" in table_names:
        family_columns = {column["name"] for column in inspector.get_columns("family_cancer_history_records")}

    with engine.begin() as connection:
        for name in (
            "bilateral_breast_cancer",
            "multiple_primary_cancers",
            "ancestry_ethnicity",
            "prior_breast_biopsy_atypia",
            "relation_degree",
        ):
            if family_columns and name not in family_columns:
                connection.execute(text(f"ALTER TABLE family_cancer_history_records ADD COLUMN {name} VARCHAR"))

    # ClinicalSummaryReview schema changes are now owned by Alembic revisions
    # under backend/migrations/. Keep this startup patcher for legacy demo
    # columns only so schema evolution has a single source of truth.
