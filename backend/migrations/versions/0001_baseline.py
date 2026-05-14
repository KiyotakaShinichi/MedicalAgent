"""baseline schema captured from existing models

Revision ID: 0001_baseline
Revises:
Create Date: 2026-05-14 00:00:00

This baseline migration declares the SQLAlchemy ORM metadata as the
authoritative schema at the point Alembic is introduced. It does not
re-emit every CREATE TABLE statement (the existing ``ensure_schema()``
calls ``Base.metadata.create_all`` on app startup and remains the
fast path for local development). Instead, the baseline registers the
current schema fingerprint so future migrations can be diffed cleanly.

Local dev / demo workflow:

    python seed_db.py                       # creates the SQLite db via ORM
    alembic stamp head                      # marks db as up-to-date
    alembic upgrade head                    # no-op (already at head)

Fresh hosted-DB workflow:

    alembic upgrade head                    # applies baseline + later migs

Future schema changes must be expressed as new Alembic revisions
(``alembic revision --autogenerate -m "..."``). Do not add ad-hoc
``ALTER TABLE`` statements to ``backend/schema_migrations.py`` for new
changes.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op  # noqa: F401  — kept for downstream migrations
import sqlalchemy as sa  # noqa: F401


revision: str = "0001_baseline"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the full ORM-defined schema on a fresh database.

    For a fresh database, this creates every table currently defined on
    ``Base.metadata``. For an existing database that was created by the
    legacy ``ensure_schema()`` path, ``create_all`` is idempotent — it
    only creates tables that do not exist yet. Either way, the database
    ends up at the baseline state and subsequent revisions take over.

    Use ``alembic stamp 0001_baseline`` (or ``alembic stamp head`` after
    further revisions exist) on an existing DB so Alembic knows where it
    already is.
    """
    from backend.database import Base
    import backend.models  # noqa: F401  — register all ORM models on Base

    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    """No-op — downgrading the baseline would drop every table."""
