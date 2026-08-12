"""
Feature importance analysis.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "traffic_volume_model.joblib"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "ml"
    / "evaluation"
    / "reports"
    / "feature_importance.png"
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


def main():

    model = joblib.load(
        MODEL_PATH
    )

    preprocessor = (
        model.named_steps[
            "preprocessor"
        ]
    )

    estimator = (
        model.named_steps[
            "model"
        ]
    )

    if not hasattr(
        estimator,
        "feature_importances_",
    ):
        print(
            "This model does not provide "
            "feature_importances_."
        )
        return

    categorical_encoder = (
        preprocessor
        .named_transformers_[
            "categorical"
        ]
    )

    categorical_names = (
        categorical_encoder
        .get_feature_names_out(
            CATEGORICAL_FEATURES
        )
    )

    feature_names = (
        NUMERICAL_FEATURES
        + list(categorical_names)
    )

    importances = (
        estimator.feature_importances_
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .head(20)
    )

    print(
        "\nTop 20 Important Features:"
    )

    print(
        importance_df
    )

    plt.figure(
        figsize=(10, 8)
    )

    sns.barplot(
        data=importance_df,
        x="importance",
        y="feature",
    )

    plt.title(
        "Top 20 Feature Importances"
    )

    plt.xlabel(
        "Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH
    )

    plt.show()


if __name__ == "__main__":
    main()