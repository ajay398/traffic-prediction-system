import pandas as pd

from ml.features.feature_engineering import (
    create_features,
)


def test_datetime_features_are_created():

    df = pd.DataFrame(
        {
            "date_time": [
                "2012-10-02 08:30:00"
            ],
            "traffic_volume": [5000],
            "temp": [280.0],
            "rain_1h": [0.0],
            "snow_1h": [0.0],
            "clouds_all": [40],
            "holiday": ["None"],
            "weather_main": ["Clear"],
            "weather_description": [
                "Sky is Clear"
            ],
        }
    )

    result = create_features(df)

    assert "year" in result.columns
    assert "month" in result.columns
    assert "hour" in result.columns
    assert "day_of_week" in result.columns


def test_weekend_feature():

    df = pd.DataFrame(
        {
            "date_time": [
                "2026-08-08 10:00:00"
            ],
            "traffic_volume": [3000],
            "temp": [280.0],
            "rain_1h": [0.0],
            "snow_1h": [0.0],
            "clouds_all": [40],
            "holiday": ["None"],
            "weather_main": ["Clear"],
            "weather_description": [
                "Sky is Clear"
            ],
        }
    )

    result = create_features(df)

    assert result["is_weekend"].iloc[0] == 1


def test_rain_flag():

    df = pd.DataFrame(
        {
            "date_time": [
                "2026-08-10 10:00:00"
            ],
            "traffic_volume": [3000],
            "temp": [280.0],
            "rain_1h": [5.0],
            "snow_1h": [0.0],
            "clouds_all": [80],
            "holiday": ["None"],
            "weather_main": ["Rain"],
            "weather_description": [
                "Light Rain"
            ],
        }
    )

    result = create_features(df)

    assert result["rain_flag"].iloc[0] == 1