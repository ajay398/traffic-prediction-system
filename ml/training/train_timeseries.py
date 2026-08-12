"""
Traffic Prediction System
Advanced Time-Series ML Training.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "processed"
    / "traffic_timeseries_features.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FEATURES
# ============================================================

NUMERICAL_FEATURES = [
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

    # Time-series features
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


CATEGORICAL_FEATURES = [
    "holiday",
    "weather_main",
    "weather_description",
    "time_period",
]


TARGET = "traffic_volume"

FEATURES = (
    NUMERICAL_FEATURES
    + CATEGORICAL_FEATURES
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(
        DATA_PATH
    )

    df["date_time"] = pd.to_datetime(
        df["date_time"]
    )

    df = df.sort_values(
        "date_time"
    ).reset_index(drop=True)

    return df


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor():

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


# ============================================================
# MODEL
# ============================================================

def create_model():

    return Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "model",
                XGBRegressor(
                    n_estimators=700,
                    learning_rate=0.03,
                    max_depth=8,
                    min_child_weight=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ADVANCED TIME-SERIES TRAFFIC MODEL")
    print("=" * 70)

    df = load_data()

    print(
        f"Dataset shape: {df.shape}"
    )

    # --------------------------------------------------------
    # Time-based split
    # --------------------------------------------------------

    split_index = int(
        len(df) * 0.80
    )

    train_df = df.iloc[
        :split_index
    ].copy()

    test_df = df.iloc[
        split_index:
    ].copy()

    print(
        f"Training rows: {len(train_df):,}"
    )

    print(
        f"Testing rows: {len(test_df):,}"
    )

    # --------------------------------------------------------
    # X / y
    # --------------------------------------------------------

    X_train = train_df[
        FEATURES
    ]

    y_train = train_df[
        TARGET
    ]

    X_test = test_df[
        FEATURES
    ]

    y_test = test_df[
        TARGET
    ]

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = create_model()

    print(
        "\nTraining advanced XGBoost..."
    )

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = (
        mean_squared_error(
            y_test,
            predictions,
        )
        ** 0.5
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    print("\nRESULTS")
    print("-" * 40)

    print(
        f"MAE  : {mae:.2f}"
    )

    print(
        f"RMSE : {rmse:.2f}"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR
        / "traffic_volume_timeseries.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        f"\nModel saved to:"
        f"\n{model_path}"
    )


if __name__ == "__main__":
    main()