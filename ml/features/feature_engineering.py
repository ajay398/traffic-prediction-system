"""
Traffic Prediction System
Feature Engineering Module.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "processed"
    / "traffic_cleaned.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "processed"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "traffic_features.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_cleaned_data() -> pd.DataFrame:
    """Load cleaned traffic data."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    return df


# ============================================================
# DATETIME FEATURES
# ============================================================

def create_datetime_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create features from date_time."""

    df = df.copy()

    df["date_time"] = pd.to_datetime(
        df["date_time"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["date_time"]
    ).copy()

    df["year"] = df["date_time"].dt.year

    df["month"] = df["date_time"].dt.month

    df["day"] = df["date_time"].dt.day

    df["hour"] = df["date_time"].dt.hour

    df["day_of_week"] = (
        df["date_time"].dt.dayofweek
    )

    df["week_of_year"] = (
        df["date_time"].dt.isocalendar().week.astype(int)
    )

    df["day_of_year"] = (
        df["date_time"].dt.dayofyear
    )

    return df


# ============================================================
# WEEKEND FEATURE
# ============================================================

def create_weekend_feature(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create weekend indicator."""

    df = df.copy()

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    return df


# ============================================================
# RUSH HOUR FEATURES
# ============================================================

def create_rush_hour_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create rush-hour related features."""

    df = df.copy()

    morning_peak_hours = [7, 8, 9]

    evening_peak_hours = [
        16,
        17,
        18,
        19,
    ]

    df["is_morning_peak"] = (
        df["hour"].isin(morning_peak_hours)
    ).astype(int)

    df["is_evening_peak"] = (
        df["hour"].isin(evening_peak_hours)
    ).astype(int)

    df["is_rush_hour"] = (
        (df["is_morning_peak"] == 1)
        | (df["is_evening_peak"] == 1)
    ).astype(int)

    return df


# ============================================================
# WEATHER FEATURES
# ============================================================

def create_weather_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create weather indicator features."""

    df = df.copy()

    df["rain_flag"] = (
        df["rain_1h"] > 0
    ).astype(int)

    df["snow_flag"] = (
        df["snow_1h"] > 0
    ).astype(int)

    return df


# ============================================================
# TIME PERIOD FEATURES
# ============================================================

def create_time_period_feature(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create broad time-of-day categories."""

    df = df.copy()

    def classify_hour(hour: int) -> str:
        if 0 <= hour < 6:
            return "night"

        if 6 <= hour < 12:
            return "morning"

        if 12 <= hour < 17:
            return "afternoon"

        if 17 <= hour < 21:
            return "evening"

        return "late_night"

    df["time_period"] = (
        df["hour"].apply(classify_hour)
    )

    return df


# ============================================================
# FEATURE PIPELINE
# ============================================================

def create_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Run the complete feature engineering pipeline."""

    df = create_datetime_features(df)

    df = create_weekend_feature(df)

    df = create_rush_hour_features(df)

    df = create_weather_features(df)

    df = create_time_period_feature(df)

    return df


# ============================================================
# SAVE FEATURES
# ============================================================

def save_features(
    df: pd.DataFrame,
) -> None:
    """Save engineered features."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Feature dataset saved to: {OUTPUT_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Run feature engineering."""

    print("=" * 60)
    print("TRAFFIC FEATURE ENGINEERING")
    print("=" * 60)

    df = load_cleaned_data()

    print(
        f"Input shape: {df.shape}"
    )

    feature_df = create_features(df)

    print(
        f"Output shape: {feature_df.shape}"
    )

    print("\nNew features:")

    original_columns = set(df.columns)

    new_columns = [
        column
        for column in feature_df.columns
        if column not in original_columns
    ]

    for column in new_columns:
        print(f"  - {column}")

    save_features(feature_df)

    print("\nFeature engineering completed.")


if __name__ == "__main__":
    main()