"""
Traffic prediction API routes.
"""

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import status
from fastapi import Depends
from sqlalchemy.orm import Session
from app.schemas.prediction import (
    PredictionHistoryItem,
    PredictionHistoryResponse,
)

from app.database.connection import get_db

from app.schemas.prediction import (
    TrafficPredictionRequest,
    TrafficPredictionResponse,
)
from app.services.prediction_service import (
    prediction_service,
)

from app.repositories.prediction_repository import (
    PredictionRepository,
)
from app.api.dependencies import (
    get_current_user,
)
from app.database.models import User


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
    db: Session = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):
    """Predict traffic volume."""

    try:

        prediction = (
            prediction_service.predict(
                db=db,
                user_id=current_user.id,
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
                prediction.predicted_traffic_volume,
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

@router.get(
    "/history",
    response_model=PredictionHistoryResponse,
)
def get_prediction_history(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):
    """Return prediction history."""

    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 100",
        )

    if offset < 0:
        raise HTTPException(
            status_code=400,
            detail="offset cannot be negative",
        )

    predictions = (
        PredictionRepository.get_history(
            db=db,
            user_id=current_user.id,
            limit=limit,
            offset=offset,
        )
    )

    total = (
    PredictionRepository.count(
        db=db,
        user_id=current_user.id,
    )
)
   


    items = [
        PredictionHistoryItem(
            id=item.id,
            prediction_time=item.prediction_time,
            predicted_traffic_volume=(
                item.predicted_traffic_volume
            ),
            model_name=item.model_name,
            created_at=item.created_at,
        )
        for item in predictions
    ]

    return PredictionHistoryResponse(
        items=items,
        total=total,
        limit=limit,
       offset=offset,
    )

@router.get(
    "/history/{prediction_id}",
    response_model=PredictionHistoryItem,
)
def get_prediction(
    prediction_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):
    """Get a prediction by ID."""

    prediction = (
        PredictionRepository.get_by_id(
            db=db,
            prediction_id=prediction_id,
            user_id=current_user.id,
        )
    )

    if prediction is None:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found",
        )

    return PredictionHistoryItem(
        id=prediction.id,
        prediction_time=prediction.prediction_time,
        predicted_traffic_volume=(
            prediction.predicted_traffic_volume
        ),
        model_name=prediction.model_name,
        created_at=prediction.created_at,
    )

@router.delete(
    "/history/{prediction_id}",
)
def delete_prediction(
    prediction_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):
    """Delete prediction history record."""

    prediction = (
        PredictionRepository.get_by_id(
            db=db,
            prediction_id=prediction_id,
            user_id=current_user.id,
        )
    )

    if prediction is None:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found",
        )

    PredictionRepository.delete(
        db=db,
        prediction=prediction,
    )

    return {
        "status": "success",
        "message": "Prediction deleted",
        "prediction_id": prediction_id,
    }
