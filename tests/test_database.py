from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.base import Base
from app.database.models import Prediction
from app.repositories.prediction_repository import (
    PredictionRepository,
)


def create_test_session():

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={
            "check_same_thread": False,
        },
    )

    Base.metadata.create_all(
        engine
    )

    Session = sessionmaker(
        bind=engine
    )

    return Session()


def test_create_prediction():

    db = create_test_session()

    prediction = Prediction(
        prediction_time=datetime.now(),
        temp=280.0,
        rain_1h=0.0,
        snow_1h=0.0,
        clouds_all=30.0,
        holiday="None",
        weather_main="Clear",
        weather_description="Clear sky",
        traffic_lag_1h=5000.0,
        traffic_lag_2h=4900.0,
        traffic_lag_3h=4800.0,
        traffic_lag_24h=5100.0,
        traffic_lag_168h=5000.0,
        rolling_mean_3h=4900.0,
        rolling_mean_6h=4850.0,
        rolling_mean_24h=4800.0,
        rolling_std_24h=300.0,
        predicted_traffic_volume=5200.0,
        model_name="test-model",
    )

    result = (
        PredictionRepository.create(
            db=db,
            prediction=prediction,
        )
    )

    assert result.id is not None

    assert (
        result.predicted_traffic_volume
        == 5200.0
    )


def test_get_prediction():

    db = create_test_session()

    prediction = Prediction(
        prediction_time=datetime.now(),
        temp=280.0,
        rain_1h=0.0,
        snow_1h=0.0,
        clouds_all=30.0,
        holiday="None",
        weather_main="Clear",
        weather_description="Clear sky",
        traffic_lag_1h=5000.0,
        traffic_lag_2h=4900.0,
        traffic_lag_3h=4800.0,
        traffic_lag_24h=5100.0,
        traffic_lag_168h=5000.0,
        rolling_mean_3h=4900.0,
        rolling_mean_6h=4850.0,
        rolling_mean_24h=4800.0,
        rolling_std_24h=300.0,
        predicted_traffic_volume=5200.0,
        model_name="test-model",
    )

    created = (
        PredictionRepository.create(
            db=db,
            prediction=prediction,
        )
    )

    result = (
        PredictionRepository.get_by_id(
            db=db,
            prediction_id=created.id,
        )
    )

    assert result is not None

    assert result.id == created.id