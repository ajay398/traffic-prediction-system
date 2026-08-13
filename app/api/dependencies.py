"""
FastAPI dependencies.
"""

from fastapi import Depends
from fastapi import HTTPException
from fastapi import status
from fastapi.security import (
    HTTPAuthorizationCredentials,
)
from fastapi.security import HTTPBearer

from app.core.security import (
    decode_access_token,
)
from app.database.connection import (
    get_db,
)
from app.database.models import User
from app.repositories.user_repository import (
    UserRepository,
)
from sqlalchemy.orm import Session


security = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = (
        Depends(security)
    ),
    db: Session = Depends(get_db),
) -> User:

    token = credentials.credentials

    try:

        payload = decode_access_token(
            token
        )

        subject = payload.get("sub")

        if subject is None:
            raise HTTPException(
                status_code=(
                    status.HTTP_401_UNAUTHORIZED
                ),
                detail="Invalid authentication token.",
            )

        user_id = int(subject)

    except (
        ValueError,
        Exception,
    ) as exc:

        raise HTTPException(
            status_code=(
                status.HTTP_401_UNAUTHORIZED
            ),
            detail="Invalid or expired token.",
        ) from exc

    user = (
        UserRepository.get_by_id(
            db,
            user_id,
        )
    )

    if user is None:
        raise HTTPException(
            status_code=(
                status.HTTP_401_UNAUTHORIZED
            ),
            detail="User not found.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=(
                status.HTTP_403_FORBIDDEN
            ),
            detail="User account is inactive.",
        )

    return user