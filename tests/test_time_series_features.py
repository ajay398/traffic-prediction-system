import pandas as pd

from ml.features.time_series_features import (
    create_time_series_features,
)


def test_lag_features():

    df = pd.DataFrame(
        {
            "date_time": pd.date_range(
                "2026-01-01",
                periods=200,
                freq="h",
            ),
            "traffic_volume": range(200),
        }
    )

    result = create_time_series_features(
        df
    )

    assert (
        "traffic_lag_1h"
        in result.columns
    )

    assert (
        "traffic_lag_24h"
        in result.columns
    )

    assert (
        "traffic_lag_168h"
        in result.columns
    )


def test_lag_uses_previous_value():

    df = pd.DataFrame(
        {
            "date_time": pd.date_range(
                "2026-01-01",
                periods=5,
                freq="h",
            ),
            "traffic_volume": [
                100,
                200,
                300,
                400,
                500,
            ],
        }
    )

    result = create_time_series_features(
        df
    )

    assert (
        result["traffic_lag_1h"].iloc[1]
        == 100
    )

    assert (
        result["traffic_lag_1h"].iloc[4]
        == 400
    )