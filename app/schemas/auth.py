"""
Authentication schemas.
"""

from datetime import datetime

from pydantic import BaseModel
from pydantic import EmailStr
from pydantic import Field


class UserRegisterRequest(BaseModel):
    """User registration request."""

    email: EmailStr

    username: str = Field(
        ...,
        min_length=3,
        max_length=100,
    )

    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
    )


class UserLoginRequest(BaseModel):
    """User login request."""

    email: EmailStr

    password: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )


class TokenResponse(BaseModel):
    """JWT response."""

    access_token: str

    token_type: str = "bearer"


class UserResponse(BaseModel):
    """Public user response."""

    id: int

    email: EmailStr

    username: str

    is_active: bool

    created_at: datetime