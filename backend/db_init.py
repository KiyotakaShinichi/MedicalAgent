from backend.database import Base, engine
from backend import models  # noqa: F401


def init_db():
    Base.metadata.create_all(bind=engine)
    print("Database initialized.")


if __name__ == "__main__":
    init_db()
