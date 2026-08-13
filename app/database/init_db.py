"""
Database initialization.
"""

from app.database.base import Base
from app.database.connection import engine

# Import models so SQLAlchemy knows about them.
from app.database.models import Prediction
from app.database.models import User # noqa: F401


def init_db() -> None:
    """Create database tables."""

    Base.metadata.create_all(
        bind=engine
    )


if __name__ == "__main__":
    init_db()

    print(
        "Database tables created successfully."
    )