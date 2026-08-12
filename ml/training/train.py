"""
Traffic Prediction System
Machine Learning Training Pipeline.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


# ============================================================
# PATH CONFIGURATION
# ============================================================

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
]


CATEGORICAL_FEATURES = [
    "holiday",
    "weather_main",
    "weather_description",
    "time_period",
]


TARGET = "traffic_volume"


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:
    """Load feature-engineered dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    df["date_time"] = pd.to_datetime(
        df["date_time"]
    )

    df = df.sort_values(
        "date_time"
    ).reset_index(drop=True)

    return df


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically.

    First 80% -> training
    Last 20%  -> testing
    """

    split_index = int(
        len(df) * 0.80
    )

    train_df = df.iloc[
        :split_index
    ].copy()

    test_df = df.iloc[
        split_index:
    ].copy()

    return train_df, test_df


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor() -> ColumnTransformer:
    """Create preprocessing pipeline."""

    preprocessor = ColumnTransformer(
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
        ],
        remainder="drop",
    )

    return preprocessor


# ============================================================
# CREATE MODELS
# ============================================================

def create_models() -> dict:
    """Create candidate ML models."""

    models = {
        "linear_regression": LinearRegression(),

        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        ),

        "xgboost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        ),
    }

    return models


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate a trained model."""

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = mean_squared_error(
        y_test,
        predictions,
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions,
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


# ============================================================
# TRAIN PIPELINE
# ============================================================

def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict, dict]:

    models = create_models()

    results = {}

    trained_models = {}

    for model_name, estimator in models.items():

        print(
            f"\nTraining: {model_name}"
        )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    create_preprocessor(),
                ),
                (
                    "model",
                    estimator,
                ),
            ]
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        metrics = evaluate_model(
            pipeline,
            X_test,
            y_test,
        )

        results[model_name] = metrics

        trained_models[model_name] = pipeline

        print(
            f"MAE  : {metrics['MAE']:.2f}"
        )

        print(
            f"RMSE : {metrics['RMSE']:.2f}"
        )

        print(
            f"R²   : {metrics['R2']:.4f}"
        )

    return results, trained_models


# ============================================================
# SELECT BEST MODEL
# ============================================================

def select_best_model(
    results: dict,
    trained_models: dict,
):
    """
    Select best model using RMSE.

    Lower RMSE is better.
    """

    best_model_name = min(
        results,
        key=lambda name: results[name]["RMSE"],
    )

    best_model = trained_models[
        best_model_name
    ]

    return best_model_name, best_model


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("TRAFFIC PREDICTION - ML TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    print(
        f"\nDataset shape: {df.shape}"
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train_df, test_df = split_data(df)

    print(
        f"Training rows: {len(train_df):,}"
    )

    print(
        f"Testing rows : {len(test_df):,}"
    )

    print(
        f"\nTraining period:"
        f"\n{train_df['date_time'].min()}"
        f"\n→ {train_df['date_time'].max()}"
    )

    print(
        f"\nTesting period:"
        f"\n{test_df['date_time'].min()}"
        f"\n→ {test_df['date_time'].max()}"
    )

    # --------------------------------------------------------
    # Prepare X and y
    # --------------------------------------------------------

    X_train = train_df[
        NUMERICAL_FEATURES
        + CATEGORICAL_FEATURES
    ]

    y_train = train_df[
        TARGET
    ]

    X_test = test_df[
        NUMERICAL_FEATURES
        + CATEGORICAL_FEATURES
    ]

    y_test = test_df[
        TARGET
    ]

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    results, trained_models = train_models(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results_df = (
        pd.DataFrame(results)
        .T
        .sort_values("RMSE")
    )

    print("\n")
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df
    )

    # --------------------------------------------------------
    # Best Model
    # --------------------------------------------------------

    best_model_name, best_model = (
        select_best_model(
            results,
            trained_models,
        )
    )

    print(
        f"\nBest model: {best_model_name}"
    )

    # --------------------------------------------------------
    # Save Best Model
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR
        / "traffic_volume_model.joblib"
    )

    joblib.dump(
        best_model,
        model_path,
    )

    print(
        f"Model saved to: {model_path}"
    )

    # --------------------------------------------------------
    # Save Results
    # --------------------------------------------------------

    results_path = (
        MODEL_DIR
        / "model_results.csv"
    )

    results_df.to_csv(
        results_path
    )

    print(
        f"Results saved to: {results_path}"
    )

    print(
        "\nTraining completed successfully."
    )


if __name__ == "__main__":
    main()