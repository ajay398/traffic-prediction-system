"""
Traffic prediction service.
"""

from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from app.core.config import settings

from sqlalchemy.orm import Session

from app.database.models import Prediction
from app.repositories.prediction_repository import (
    PredictionRepository,
)


class PredictionService:
    """Service responsible for loading and running the ML model."""

    def __init__(self):
        self.model = None
        self.model_path = Path(
            settings.model_path
        )

    def load_model(self) -> None:
        """Load the trained model."""

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        self.model = joblib.load(
            self.model_path
        )

    def is_loaded(self) -> bool:
        """Check whether model is loaded."""

        return self.model is not None

    def predict(
        self,
        *,
        db: Session,
        user_id: int,
        date_time: datetime,
        temp: float,
        rain_1h: float,
        snow_1h: float,
        clouds_all: float,
        holiday: str,
        weather_main: str,
        weather_description: str,
        traffic_lag_1h: float,
        traffic_lag_2h: float,
        traffic_lag_3h: float,
        traffic_lag_24h: float,
        traffic_lag_168h: float,
        rolling_mean_3h: float,
        rolling_mean_6h: float,
        rolling_mean_24h: float,
        rolling_std_24h: float,
    ) -> float:
        """Generate traffic prediction."""

        if self.model is None:
            raise RuntimeError(
                "Prediction model is not loaded."
            )

        hour = date_time.hour

        month = date_time.month

        day = date_time.day

        year = date_time.year

        day_of_week = date_time.weekday()

        week_of_year = (
            date_time.isocalendar().week
        )

        day_of_year = (
            date_time.timetuple().tm_yday
        )

        is_weekend = int(
            day_of_week >= 5
        )

        is_morning_peak = int(
            hour in [7, 8, 9]
        )

        is_evening_peak = int(
            hour in [16, 17, 18, 19]
        )

        is_rush_hour = int(
            is_morning_peak
            or is_evening_peak
        )

        rain_flag = int(
            rain_1h > 0
        )

        snow_flag = int(
            snow_1h > 0
        )

        if 0 <= hour < 6:
            time_period = "night"

        elif 6 <= hour < 12:
            time_period = "morning"

        elif 12 <= hour < 17:
            time_period = "afternoon"

        elif 17 <= hour < 21:
            time_period = "evening"

        else:
            time_period = "late_night"

        data = {
            "temp": temp,
            "rain_1h": rain_1h,
            "snow_1h": snow_1h,
            "clouds_all": clouds_all,

            "year": year,
            "month": month,
            "day": day,
            "hour": hour,
            "day_of_week": day_of_week,
            "week_of_year": week_of_year,
            "day_of_year": day_of_year,

            "is_weekend": is_weekend,
            "is_morning_peak": is_morning_peak,
            "is_evening_peak": is_evening_peak,
            "is_rush_hour": is_rush_hour,

            "rain_flag": rain_flag,
            "snow_flag": snow_flag,

            "traffic_lag_1h": traffic_lag_1h,
            "traffic_lag_2h": traffic_lag_2h,
            "traffic_lag_3h": traffic_lag_3h,
            "traffic_lag_24h": traffic_lag_24h,
            "traffic_lag_168h": traffic_lag_168h,

            "rolling_mean_3h": rolling_mean_3h,
            "rolling_mean_6h": rolling_mean_6h,
            "rolling_mean_24h": rolling_mean_24h,
            "rolling_std_24h": rolling_std_24h,

            "holiday": holiday,
            "weather_main": weather_main,
            "weather_description": weather_description,
            "time_period": time_period,
        }

        input_df = pd.DataFrame(
            [data]
        )

        prediction = self.model.predict(
            input_df
        )

        predicted_value = float(
        prediction[0]
        )

        prediction_record = Prediction(
        user_id=user_id,
        prediction_time=date_time,
        temp=temp,
        rain_1h=rain_1h,
        snow_1h=snow_1h,
        clouds_all=clouds_all,
        holiday=holiday,
        weather_main=weather_main,
        weather_description=weather_description,
        traffic_lag_1h=traffic_lag_1h,
        traffic_lag_2h=traffic_lag_2h,
        traffic_lag_3h=traffic_lag_3h,
        traffic_lag_24h=traffic_lag_24h,
        traffic_lag_168h=traffic_lag_168h,
        rolling_mean_3h=rolling_mean_3h,
        rolling_mean_6h=rolling_mean_6h,
        rolling_mean_24h=rolling_mean_24h,
        rolling_std_24h=rolling_std_24h,
        predicted_traffic_volume=predicted_value,
        model_name="xgboost-timeseries",
)

        return PredictionRepository.create(
    db=db,
    prediction=prediction_record,
)


prediction_service = PredictionService()
