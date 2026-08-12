"""
Traffic Prediction System
Time-Series Feature Engineering.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "processed"
    / "traffic_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "processed"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "traffic_timeseries_features.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:
    """Load feature-engineered traffic data."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    df["date_time"] = pd.to_datetime(
        df["date_time"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["date_time"]
    ).copy()

    df = df.sort_values(
        "date_time"
    ).reset_index(drop=True)

    return df


# ============================================================
# LAG FEATURES
# ============================================================

def create_lag_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create historical traffic features."""

    df = df.copy()

    # Previous hour
    df["traffic_lag_1h"] = (
        df["traffic_volume"].shift(1)
    )

    # Two hours ago
    df["traffic_lag_2h"] = (
        df["traffic_volume"].shift(2)
    )

    # Three hours ago
    df["traffic_lag_3h"] = (
        df["traffic_volume"].shift(3)
    )

    # Same hour previous day
    df["traffic_lag_24h"] = (
        df["traffic_volume"].shift(24)
    )

    # Same hour previous week
    df["traffic_lag_168h"] = (
        df["traffic_volume"].shift(168)
    )

    return df


# ============================================================
# ROLLING FEATURES
# ============================================================

def create_rolling_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create historical rolling traffic statistics."""

    df = df.copy()

    # Shift first so current target is NOT included.
    previous_traffic = (
        df["traffic_volume"].shift(1)
    )

    df["rolling_mean_3h"] = (
        previous_traffic
        .rolling(window=3)
        .mean()
    )

    df["rolling_mean_6h"] = (
        previous_traffic
        .rolling(window=6)
        .mean()
    )

    df["rolling_mean_24h"] = (
        previous_traffic
        .rolling(window=24)
        .mean()
    )

    return df


# ============================================================
# ROLLING STANDARD DEVIATION
# ============================================================

def create_rolling_std_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create rolling traffic volatility features."""

    df = df.copy()

    previous_traffic = (
        df["traffic_volume"].shift(1)
    )

    df["rolling_std_24h"] = (
        previous_traffic
        .rolling(window=24)
        .std()
    )

    return df


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def create_time_series_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create all time-series features."""

    df = create_lag_features(df)

    df = create_rolling_features(df)

    df = create_rolling_std_features(df)

    return df


# ============================================================
# SAVE
# ============================================================

def save_data(
    df: pd.DataFrame,
) -> None:
    """Save time-series feature dataset."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Saved dataset to: {OUTPUT_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TIME-SERIES FEATURE ENGINEERING")
    print("=" * 70)

    df = load_data()

    print(
        f"Original shape: {df.shape}"
    )

    df = create_time_series_features(df)

    print(
        f"After feature creation: {df.shape}"
    )

    new_columns = [
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

    print("\nCreated features:")

    for column in new_columns:
        print(f"  - {column}")

    save_data(df)


if __name__ == "__main__":
    main()