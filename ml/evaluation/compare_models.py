"""
Compare baseline and time-series traffic models.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASELINE_DATA = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "processed"
    / "traffic_features.csv"
)

TIMESERIES_DATA = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "processed"
    / "traffic_timeseries_features.csv"
)

BASELINE_MODEL = (
    PROJECT_ROOT
    / "models"
    / "traffic_volume_model.joblib"
)

TIMESERIES_MODEL = (
    PROJECT_ROOT
    / "models"
    / "traffic_volume_timeseries.joblib"
)


BASELINE_FEATURES = [
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "year",
    "month",
    "day",
    "hour",
    "day_of_week",
    "week_of_year",
    "day_of_year",
    "is_weekend",
    "is_morning_peak",
    "is_evening_peak",
    "is_rush_hour",
    "rain_flag",
    "snow_flag",
    "holiday",
    "weather_main",
    "weather_description",
    "time_period",
]


TIMESERIES_NUMERIC_FEATURES = [
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "year",
    "month",
    "day",
    "hour",
    "day_of_week",
    "week_of_year",
    "day_of_year",
    "is_weekend",
    "is_morning_peak",
    "is_evening_peak",
    "is_rush_hour",
    "rain_flag",
    "snow_flag",
    "traffic_lag_1h",
    "traffic_lag_2h",
    "traffic_lag_3h",
    "traffic_lag_24h",
    "traffic_lag_168h",
    "rolling_mean_3h",
    "rolling_mean_6h",
    "rolling_mean_24h",
    "rolling_std_24h",
]


TIMESERIES_FEATURES = (
    TIMESERIES_NUMERIC_FEATURES
    + [
        "holiday",
        "weather_main",
        "weather_description",
        "time_period",
    ]
)


TARGET = "traffic_volume"


def evaluate(
    model,
    X,
    y,
):
    predictions = model.predict(X)

    return {
        "MAE": mean_absolute_error(
            y,
            predictions,
        ),
        "RMSE": (
            mean_squared_error(
                y,
                predictions,
            )
            ** 0.5
        ),
        "R2": r2_score(
            y,
            predictions,
        ),
    }


def main():

    baseline_df = pd.read_csv(
        BASELINE_DATA
    )

    timeseries_df = pd.read_csv(
        TIMESERIES_DATA
    )

    baseline_df["date_time"] = (
        pd.to_datetime(
            baseline_df["date_time"]
        )
    )

    timeseries_df["date_time"] = (
        pd.to_datetime(
            timeseries_df["date_time"]
        )
    )

    # --------------------------------------------------------
    # Use comparable future test periods
    # --------------------------------------------------------

    baseline_split = int(
        len(baseline_df) * 0.80
    )

    timeseries_split = int(
        len(timeseries_df) * 0.80
    )

    baseline_test = (
        baseline_df
        .iloc[baseline_split:]
        .copy()
    )

    timeseries_test = (
        timeseries_df
        .iloc[timeseries_split:]
        .copy()
    )

    baseline_model = joblib.load(
        BASELINE_MODEL
    )

    timeseries_model = joblib.load(
        TIMESERIES_MODEL
    )

    baseline_results = evaluate(
        baseline_model,
        baseline_test[
            BASELINE_FEATURES
        ],
        baseline_test[TARGET],
    )

    timeseries_results = evaluate(
        timeseries_model,
        timeseries_test[
            TIMESERIES_FEATURES
        ],
        timeseries_test[TARGET],
    )

    results = pd.DataFrame(
        [
            baseline_results,
            timeseries_results,
        ],
        index=[
            "Baseline",
            "Time-Series",
        ],
    )

    print(
        "\nMODEL COMPARISON"
    )

    print(
        results
    )

    results.to_csv(
        PROJECT_ROOT
        / "models"
        / "model_comparison.csv"
    )


if __name__ == "__main__":
    main()