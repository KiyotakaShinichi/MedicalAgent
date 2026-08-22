"""Shared helpers scoped to the breast-monitoring integration suite."""

import uuid
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from tests.breast_monitoring.environment import (
    configure_breast_monitoring_test_environment,
)

configure_breast_monitoring_test_environment()

from backend.database import Base  # noqa: E402


class FakeSeries:
    def __init__(self, role, description, instances):
        self.candidate_role = role
        self.series_description = description
        self.instance_count = instances


def _temp_db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return Session()


def _temp_root():
    test_root = Path("Data/test_tmp")
    test_root.mkdir(parents=True, exist_ok=True)
    return test_root


def _make_temp_dir(root):
    path = Path(root) / f"unit_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path
