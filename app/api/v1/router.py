"""
API v1 router.
"""

from fastapi import APIRouter

from app.api.v1.auth import (
    router as auth_router,
)
from app.api.v1.predictions import (
    router as predictions_router,
)


api_router = APIRouter(
    prefix="/api/v1"
)


api_router.include_router(
    auth_router
)

api_router.include_router(
    predictions_router
)