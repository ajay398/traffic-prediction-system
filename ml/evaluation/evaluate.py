"""
Traffic Prediction System
Model Evaluation Module.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "processed"
    / "traffic_features.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "traffic_volume_model.joblib"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "evaluation"
    / "reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load feature dataset."""

    df = pd.read_csv(DATA_PATH)

    df["date_time"] = pd.to_datetime(
        df["date_time"]
    )

    df = df.sort_values(
        "date_time"
    ).reset_index(drop=True)

    return df


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


FEATURES = (
    NUMERICAL_FEATURES
    + CATEGORICAL_FEATURES
)


TARGET = "traffic_volume"


# ============================================================
# TIME-BASED TEST SET
# ============================================================

def get_test_data(df):

    split_index = int(
        len(df) * 0.80
    )

    test_df = df.iloc[
        split_index:
    ].copy()

    X_test = test_df[FEATURES]

    y_test = test_df[TARGET]

    return test_df, X_test, y_test


# ============================================================
# PREDICTIONS
# ============================================================

def generate_predictions():

    df = load_data()

    test_df, X_test, y_test = (
        get_test_data(df)
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    predictions = model.predict(
        X_test
    )

    results = test_df[
        [
            "date_time",
            "traffic_volume",
            "hour",
            "day_of_week",
            "weather_main",
        ]
    ].copy()

    results["prediction"] = predictions

    results["error"] = (
        results["traffic_volume"]
        - results["prediction"]
    )

    results["absolute_error"] = (
        results["error"].abs()
    )

    return model, results, y_test


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(results):

    actual = results[
        "traffic_volume"
    ]

    predicted = results[
        "prediction"
    ]

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    rmse = mean_squared_error(
        actual,
        predicted,
    ) ** 0.5

    r2 = r2_score(
        actual,
        predicted,
    )

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }

    return metrics


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

def plot_actual_vs_predicted(results):

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=results,
        x="traffic_volume",
        y="prediction",
        alpha=0.4,
    )

    minimum = min(
        results["traffic_volume"].min(),
        results["prediction"].min(),
    )

    maximum = max(
        results["traffic_volume"].max(),
        results["prediction"].max(),
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
    )

    plt.title(
        "Actual vs Predicted Traffic Volume"
    )

    plt.xlabel(
        "Actual Traffic Volume"
    )

    plt.ylabel(
        "Predicted Traffic Volume"
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR
        / "actual_vs_predicted.png"
    )

    plt.show()


# ============================================================
# RESIDUAL DISTRIBUTION
# ============================================================

def plot_residuals(results):

    plt.figure(figsize=(10, 6))

    sns.histplot(
        results["error"],
        bins=50,
        kde=True,
    )

    plt.axvline(
        0,
        linestyle="--",
    )

    plt.title(
        "Prediction Error Distribution"
    )

    plt.xlabel(
        "Residual"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR
        / "residual_distribution.png"
    )

    plt.show()


# ============================================================
# ERROR BY HOUR
# ============================================================

def analyze_error_by_hour(results):

    hourly_error = (
        results
        .groupby("hour")[
            "absolute_error"
        ]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(12, 6))

    sns.lineplot(
        data=hourly_error,
        x="hour",
        y="absolute_error",
        marker="o",
    )

    plt.title(
        "Average Prediction Error by Hour"
    )

    plt.xlabel(
        "Hour"
    )

    plt.ylabel(
        "Mean Absolute Error"
    )

    plt.xticks(range(24))

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR
        / "error_by_hour.png"
    )

    plt.show()

    return hourly_error


# ============================================================
# ERROR BY WEATHER
# ============================================================

def analyze_error_by_weather(results):

    weather_error = (
        results
        .groupby("weather_main")[
            "absolute_error"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
        .reset_index()
    )

    print(
        "\nError by Weather:"
    )

    print(weather_error)

    return weather_error


# ============================================================
# WORST PREDICTIONS
# ============================================================

def show_worst_predictions(
    results,
    n=20,
):

    worst = (
        results
        .sort_values(
            "absolute_error",
            ascending=False,
        )
        .head(n)
    )

    print(
        "\nWorst Predictions:"
    )

    print(worst)

    return worst


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TRAFFIC MODEL EVALUATION")
    print("=" * 70)

    model, results, y_test = (
        generate_predictions()
    )

    metrics = calculate_metrics(
        results
    )

    print("\nMODEL METRICS")
    print("-" * 40)

    for name, value in metrics.items():

        if name == "R2":
            print(
                f"{name}: {value:.4f}"
            )
        else:
            print(
                f"{name}: {value:.2f}"
            )

    results.to_csv(
        REPORT_DIR
        / "predictions.csv",
        index=False,
    )

    plot_actual_vs_predicted(
        results
    )

    plot_residuals(
        results
    )

    hourly_error = (
        analyze_error_by_hour(
            results
        )
    )

    weather_error = (
        analyze_error_by_weather(
            results
        )
    )

    show_worst_predictions(
        results
    )

    print(
        "\nEvaluation completed."
    )


if __name__ == "__main__":
    main()