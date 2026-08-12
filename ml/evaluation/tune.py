"""
Traffic Prediction System
XGBoost Hyperparameter Tuning.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "processed"
    / "traffic_features.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

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
]

CATEGORICAL_FEATURES = [
    "holiday",
    "weather_main",
    "weather_description",
    "time_period",
]

FEATURES = (
    NUMERICAL_FEATURES
    + CATEGORICAL_FEATURES
)

TARGET = "traffic_volume"


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


def create_model(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    min_child_weight=1,
):

    return Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "model",
                XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    min_child_weight=min_child_weight,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main():

    print("=" * 70)
    print("XGBOOST MODEL IMPROVEMENT")
    print("=" * 70)

    df = load_data()

    split_index = int(
        len(df) * 0.80
    )

    train_df = df.iloc[
        :split_index
    ].copy()

    test_df = df.iloc[
        split_index:
    ].copy()

    X_train = train_df[FEATURES]

    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]

    y_test = test_df[TARGET]

    # --------------------------------------------------------
    # TimeSeriesSplit
    # --------------------------------------------------------

    tscv = TimeSeriesSplit(
        n_splits=3
    )

    parameter_sets = [
        {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "min_child_weight": 1,
        },
        {
            "n_estimators": 500,
            "max_depth": 8,
            "learning_rate": 0.05,
            "min_child_weight": 1,
        },
        {
            "n_estimators": 700,
            "max_depth": 8,
            "learning_rate": 0.03,
            "min_child_weight": 3,
        },
        {
            "n_estimators": 500,
            "max_depth": 10,
            "learning_rate": 0.03,
            "min_child_weight": 3,
        },
    ]

    best_params = None
    best_cv_rmse = float("inf")

    print(
        "\nStarting time-series validation..."
    )

    for params in parameter_sets:

        fold_scores = []

        print(
            f"\nTesting parameters: {params}"
        )

        for fold, (
            train_index,
            validation_index,
        ) in enumerate(
            tscv.split(X_train),
            start=1,
        ):

            X_fold_train = (
                X_train.iloc[train_index]
            )

            y_fold_train = (
                y_train.iloc[train_index]
            )

            X_fold_val = (
                X_train.iloc[validation_index]
            )

            y_fold_val = (
                y_train.iloc[validation_index]
            )

            model = create_model(
                **params
            )

            model.fit(
                X_fold_train,
                y_fold_train,
            )

            predictions = model.predict(
                X_fold_val
            )

            rmse = (
                mean_squared_error(
                    y_fold_val,
                    predictions,
                )
                ** 0.5
            )

            fold_scores.append(
                rmse
            )

            print(
                f"Fold {fold}: "
                f"RMSE = {rmse:.2f}"
            )

        average_rmse = sum(
            fold_scores
        ) / len(fold_scores)

        print(
            f"Average CV RMSE: "
            f"{average_rmse:.2f}"
        )

        if average_rmse < best_cv_rmse:

            best_cv_rmse = (
                average_rmse
            )

            best_params = params

    # --------------------------------------------------------
    # Train final model
    # --------------------------------------------------------

    print(
        "\nBest parameters:"
    )

    print(
        best_params
    )

    print(
        f"\nBest CV RMSE: "
        f"{best_cv_rmse:.2f}"
    )

    final_model = create_model(
        **best_params
    )

    final_model.fit(
        X_train,
        y_train,
    )

    predictions = final_model.predict(
        X_test
    )

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

    print("\nFINAL TUNED MODEL")
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

    model_path = (
        MODEL_DIR
        / "traffic_volume_model_tuned.joblib"
    )

    joblib.dump(
        final_model,
        model_path,
    )

    print(
        f"\nTuned model saved to:"
        f"\n{model_path}"
    )


if __name__ == "__main__":
    main()