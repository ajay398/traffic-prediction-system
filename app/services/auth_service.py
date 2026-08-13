"""
Authentication service.
"""

from sqlalchemy.orm import Session

from app.core.security import (
    create_access_token,
    hash_password,
    verify_password,
)
from app.database.models import User
from app.repositories.user_repository import (
    UserRepository,
)


class AuthService:
    """Handle registration and authentication."""

    @staticmethod
    def register(
        db: Session,
        email: str,
        username: str,
        password: str,
    ) -> User:

        existing_email = (
            UserRepository.get_by_email(
                db,
                email,
            )
        )

        if existing_email:
            raise ValueError(
                "Email already registered."
            )

        existing_username = (
            UserRepository.get_by_username(
                db,
                username,
            )
        )

        if existing_username:
            raise ValueError(
                "Username already registered."
            )

        user = User(
            email=email,
            username=username,
            hashed_password=hash_password(
                password
            ),
            is_active=True,
        )

        return UserRepository.create(
            db,
            user,
        )

    @staticmethod
    def authenticate(
        db: Session,
        email: str,
        password: str,
    ) -> User | None:

        user = (
            UserRepository.get_by_email(
                db,
                email,
            )
        )

        if user is None:
            return None

        if not verify_password(
            password,
            user.hashed_password,
        ):
            return None

        if not user.is_active:
            return None

        return user

    @staticmethod
    def create_token(
        user: User,
    ) -> str:

        return create_access_token(
            user.id
        )