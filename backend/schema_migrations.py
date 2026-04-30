from sqlalchemy import inspect, text

from backend.database import Base, engine


def ensure_schema():
    Base.metadata.create_all(bind=engine)

    inspector = inspect(engine)
    lab_columns = {column["name"] for column in inspector.get_columns("lab_results")}

    with engine.begin() as connection:
        if "source" not in lab_columns:
            connection.execute(text("ALTER TABLE lab_results ADD COLUMN source VARCHAR DEFAULT 'manual' NOT NULL"))
        if "source_note" not in lab_columns:
            connection.execute(text("ALTER TABLE lab_results ADD COLUMN source_note TEXT"))
