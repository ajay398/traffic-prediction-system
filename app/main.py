"""
Traffic Prediction FastAPI application.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.router import api_router
from app.core.config import settings
from app.services.prediction_service import (
    prediction_service,
)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    """Application startup and shutdown."""

    print(
        "Loading traffic prediction model..."
    )

    prediction_service.load_model()

    print(
        "Traffic prediction model loaded."
    )

    yield

    print(
        "Application shutting down."
    )


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Production-style API for "
        "traffic volume prediction."
    ),
    lifespan=lifespan,
)


app.include_router(
    api_router
)


@app.get(
    "/api/v1/health",
    tags=["Health"],
)
def health_check():
    """API health check."""

    return {
        "status": "healthy",
        "model_loaded": (
            prediction_service.is_loaded()
        ),
    }


@app.get(
    "/api/v1/model-info",
    tags=["Health"],
)
def model_info():
    """Return model information."""

    return {
        "model": "XGBoost",
        "type": "time-series regression",
        "features": [
            "weather",
            "calendar",
            "traffic_lags",
            "rolling_statistics",
        ],
        "status": (
            "loaded"
            if prediction_service.is_loaded()
            else "not_loaded"
        ),
    }