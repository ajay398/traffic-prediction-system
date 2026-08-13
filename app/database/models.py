"""
Database models.
"""

from datetime import datetime

from sqlalchemy import DateTime
from sqlalchemy import Float
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import func
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column

from app.database.base import Base


class Prediction(Base):
    """
    Stores traffic prediction requests
    and their results.
    """

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    prediction_time: Mapped[datetime] = (
        mapped_column(
            DateTime,
            nullable=False,
        )
    )

    temp: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    rain_1h: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )

    snow_1h: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )

    clouds_all: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    holiday: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    weather_main: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    weather_description: Mapped[str] = (
        mapped_column(
            String(255),
            nullable=False,
        )
    )

    traffic_lag_1h: Mapped[float] = (
        mapped_column(
            Float,
            nullable=False,
        )
    )

    traffic_lag_2h: Mapped[float] = (
        mapped_column(
            Float,
            nullable=False,
        )
    )

    traffic_lag_3h: Mapped[float] = (
        mapped_column(
            Float,
            nullable=False,
        )
    )

    traffic_lag_24h: Mapped[float] = (
        mapped_column(
            Float,
            nullable=False,
        )
    )

    traffic_lag_168h: Mapped[float] = (
        mapped_column(
            Float,
            nullable=False,
        )
    )

    rolling_mean_3h: Mapped[float] = (
        mapped_column(
            Float,
            nullable=False,
        )
    )

    rolling_mean_6h: Mapped[float] = (
        mapped_column(
            Float,
            nullable=False,
        )
    )

    rolling_mean_24h: Mapped[float] = (
        mapped_column(
            Float,
            nullable=False,
        )
    )

    rolling_std_24h: Mapped[float] = (
        mapped_column(
            Float,
            nullable=False,
        )
    )

    predicted_traffic_volume: Mapped[float] = (
        mapped_column(
            Float,
            nullable=False,
        )
    )

    model_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    created_at: Mapped[datetime] = (
        mapped_column(
            DateTime,
            server_default=func.now(),
            nullable=False,
        )
    )