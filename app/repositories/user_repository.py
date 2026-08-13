"""
User database repository.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import User


class UserRepository:
    """Database operations for users."""

    @staticmethod
    def get_by_email(
        db: Session,
        email: str,
    ) -> User | None:
        """Find user by email."""

        statement = select(
            User
        ).where(
            User.email == email
        )

        return db.scalar(statement)

    @staticmethod
    def get_by_username(
        db: Session,
        username: str,
    ) -> User | None:
        """Find user by username."""

        statement = select(
            User
        ).where(
            User.username == username
        )

        return db.scalar(statement)

    @staticmethod
    def get_by_id(
        db: Session,
        user_id: int,
    ) -> User | None:
        """Find user by ID."""

        statement = select(
            User
        ).where(
            User.id == user_id
        )

        return db.scalar(statement)

    @staticmethod
    def create(
        db: Session,
        user: User,
    ) -> User:
        """Create user."""

        db.add(user)

        db.commit()

        db.refresh(user)

        return user