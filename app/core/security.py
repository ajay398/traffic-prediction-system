"""
Authentication and password security utilities.
"""

from datetime import datetime
from datetime import timedelta
from datetime import timezone

import jwt

from pwdlib import PasswordHash

from app.core.config import settings


password_hash = PasswordHash.recommended()


def hash_password(
    password: str,
) -> str:
    """Hash a password securely."""

    return password_hash.hash(
        password
    )


def verify_password(
    plain_password: str,
    hashed_password: str,
) -> bool:
    """Verify a password against its hash."""

    return password_hash.verify(
        plain_password,
        hashed_password,
    )


def create_access_token(
    user_id: int,
) -> str:
    """Create JWT access token."""

    expires_delta = timedelta(
        minutes=(
            settings.access_token_expire_minutes
        )
    )

    expire = (
        datetime.now(timezone.utc)
        + expires_delta
    )

    payload = {
        "sub": str(user_id),
        "exp": expire,
    }

    return jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )


def decode_access_token(
    token: str,
) -> dict:
    """Decode and validate JWT token."""

    return jwt.decode(
        token,
        settings.jwt_secret_key,
        algorithms=[
            settings.jwt_algorithm
        ],
    )