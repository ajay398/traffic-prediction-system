"""
Traffic prediction API routes.
"""

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import status

from app.schemas.prediction import (
    TrafficPredictionRequest,
    TrafficPredictionResponse,
)
from app.services.prediction_service import (
    prediction_service,
)


router = APIRouter(
    prefix="/predictions",
    tags=["Predictions"],
)


@router.post(
    "/traffic",
    response_model=TrafficPredictionResponse,
)
def predict_traffic(
    request: TrafficPredictionRequest,
):
    """Predict traffic volume."""

    try:

        prediction = (
            prediction_service.predict(
                date_time=request.date_time,
                temp=request.temp,
                rain_1h=request.rain_1h,
                snow_1h=request.snow_1h,
                clouds_all=request.clouds_all,
                holiday=request.holiday,
                weather_main=request.weather_main,
                weather_description=(
                    request.weather_description
                ),
                traffic_lag_1h=(
                    request.traffic_lag_1h
                ),
                traffic_lag_2h=(
                    request.traffic_lag_2h
                ),
                traffic_lag_3h=(
                    request.traffic_lag_3h
                ),
                traffic_lag_24h=(
                    request.traffic_lag_24h
                ),
                traffic_lag_168h=(
                    request.traffic_lag_168h
                ),
                rolling_mean_3h=(
                    request.rolling_mean_3h
                ),
                rolling_mean_6h=(
                    request.rolling_mean_6h
                ),
                rolling_mean_24h=(
                    request.rolling_mean_24h
                ),
                rolling_std_24h=(
                    request.rolling_std_24h
                ),
            )
        )

        return TrafficPredictionResponse(
            predicted_traffic_volume=round(
                prediction,
                2,
            ),
            model=(
                "xgboost-timeseries"
            ),
            status="success",
        )

    except Exception as exc:

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=str(exc),
        ) from exc