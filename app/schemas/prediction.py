"""
Prediction request and response schemas.
"""

from datetime import datetime

from pydantic import BaseModel
from pydantic import Field


class TrafficPredictionRequest(BaseModel):
    """Traffic prediction request."""

    date_time: datetime

    temp: float = Field(
        ...,
        description="Temperature.",
    )

    rain_1h: float = Field(
        0.0,
        ge=0,
        description="Rainfall during the previous hour.",
    )

    snow_1h: float = Field(
        0.0,
        ge=0,
        description="Snowfall during the previous hour.",
    )

    clouds_all: float = Field(
        ...,
        ge=0,
        le=100,
        description="Cloud coverage percentage.",
    )

    holiday: str = Field(
        "None",
        description="Holiday name or None.",
    )

    weather_main: str = Field(
        ...,
        description="Main weather category.",
    )

    weather_description: str = Field(
        ...,
        description="Detailed weather description.",
    )

    traffic_lag_1h: float = Field(
        ...,
        ge=0,
    )

    traffic_lag_2h: float = Field(
        ...,
        ge=0,
    )

    traffic_lag_3h: float = Field(
        ...,
        ge=0,
    )

    traffic_lag_24h: float = Field(
        ...,
        ge=0,
    )

    traffic_lag_168h: float = Field(
        ...,
        ge=0,
    )

    rolling_mean_3h: float = Field(
        ...,
        ge=0,
    )

    rolling_mean_6h: float = Field(
        ...,
        ge=0,
    )

    rolling_mean_24h: float = Field(
        ...,
        ge=0,
    )

    rolling_std_24h: float = Field(
        0.0,
        ge=0,
    )


class TrafficPredictionResponse(BaseModel):
    """Traffic prediction response."""

    predicted_traffic_volume: float

    model: str

    status: str


class PredictionHistoryItem(BaseModel):
    """Prediction history response."""

    id: int

    prediction_time: datetime

    predicted_traffic_volume: float

    model_name: str

    created_at: datetime


class PredictionHistoryResponse(BaseModel):
    """Paginated prediction history."""

    items: list[PredictionHistoryItem]

    total: int

    limit: int

    offset: int

