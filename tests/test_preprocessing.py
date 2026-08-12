import pandas as pd

from ml.preprocessing.preprocess import preprocess_data


def test_preprocessing_removes_duplicates():
    df = pd.DataFrame(
        {
            "holiday": ["None", "None"],
            "temp": [280.0, 280.0],
            "rain_1h": [0.0, 0.0],
            "snow_1h": [0.0, 0.0],
            "clouds_all": [40, 40],
            "weather_main": ["Clear", "Clear"],
            "weather_description": [
                "Sky is Clear",
                "Sky is Clear",
            ],
            "date_time": [
                "2012-10-02 09:00:00",
                "2012-10-02 09:00:00",
            ],
            "traffic_volume": [5000, 5000],
        }
    )

    result = preprocess_data(df)

    assert len(result) == 1


def test_preprocessing_removes_negative_traffic():
    df = pd.DataFrame(
        {
            "holiday": ["None", "None"],
            "temp": [280.0, 280.0],
            "rain_1h": [0.0, 0.0],
            "snow_1h": [0.0, 0.0],
            "clouds_all": [40, 40],
            "weather_main": ["Clear", "Clear"],
            "weather_description": [
                "Sky is Clear",
                "Sky is Clear",
            ],
            "date_time": [
                "2012-10-02 09:00:00",
                "2012-10-02 10:00:00",
            ],
            "traffic_volume": [5000, -100],
        }
    )

    result = preprocess_data(df)

    assert (result["traffic_volume"] >= 0).all()