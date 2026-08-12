"""
Traffic Prediction System
Data Cleaning and Preprocessing Pipeline.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "raw"
    / "Metro_Interstate_Traffic_Volume.csv"
)

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "processed"
)

PROCESSED_DATA_PATH = (
    PROCESSED_DATA_DIR
    / "traffic_cleaned.csv"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = {
    "holiday",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "weather_main",
    "weather_description",
    "date_time",
    "traffic_volume",
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:
    """Load the raw traffic dataset."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Loaded dataset: {df.shape}")

    return df


# ============================================================
# VALIDATE COLUMNS
# ============================================================

def validate_columns(df: pd.DataFrame) -> None:
    """Validate that all required columns exist."""

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("Column validation successful.")


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names."""

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return df


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate records."""

    before = len(df)

    df = df.drop_duplicates().copy()

    after = len(df)

    removed = before - after

    print(f"Duplicates removed: {removed}")

    return df


# ============================================================
# CLEAN DATETIME
# ============================================================

def clean_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Convert date_time to pandas datetime."""

    df = df.copy()

    df["date_time"] = pd.to_datetime(
        df["date_time"],
        errors="coerce"
    )

    invalid_dates = df["date_time"].isna().sum()

    print(f"Invalid datetime values: {invalid_dates}")

    df = df.dropna(
        subset=["date_time"]
    ).copy()

    return df


# ============================================================
# CLEAN NUMERIC COLUMNS
# ============================================================

def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric columns to numeric types."""

    df = df.copy()

    numeric_columns = [
        "temp",
        "rain_1h",
        "snow_1h",
        "clouds_all",
        "traffic_volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    return df


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values."""

    df = df.copy()

    numeric_columns = [
        "temp",
        "rain_1h",
        "snow_1h",
        "clouds_all",
    ]

    categorical_columns = [
        "holiday",
        "weather_main",
        "weather_description",
    ]

    for column in numeric_columns:
        df[column] = df[column].fillna(
            df[column].median()
        )

    for column in categorical_columns:
        df[column] = df[column].fillna(
            "Unknown"
        )

    # Target should not be imputed.
    # Rows without a target cannot be used for supervised learning.
    df = df.dropna(
        subset=["traffic_volume"]
    ).copy()

    return df


# ============================================================
# CLEAN CATEGORICAL VALUES
# ============================================================

def clean_categorical_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Clean categorical text columns."""

    df = df.copy()

    categorical_columns = [
        "holiday",
        "weather_main",
        "weather_description",
    ]

    for column in categorical_columns:
        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    return df


# ============================================================
# HANDLE INVALID NUMERIC VALUES
# ============================================================

def handle_invalid_values(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Handle physically invalid values."""

    df = df.copy()

    # Traffic volume cannot be negative.
    df = df[
        df["traffic_volume"] >= 0
    ].copy()

    # Rainfall cannot be negative.
    df = df[
        df["rain_1h"] >= 0
    ].copy()

    # Snowfall cannot be negative.
    df = df[
        df["snow_1h"] >= 0
    ].copy()

    # Cloud coverage should be between 0 and 100.
    df = df[
        df["clouds_all"].between(0, 100)
    ].copy()

    return df


# ============================================================
# SORT DATA
# ============================================================

def sort_by_datetime(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Sort observations chronologically."""

    df = df.sort_values(
        "date_time"
    ).reset_index(drop=True)

    return df


# ============================================================
# SAVE DATA
# ============================================================

def save_data(df: pd.DataFrame) -> None:
    """Save cleaned dataset."""

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )

    print(
        f"Cleaned dataset saved to: "
        f"{PROCESSED_DATA_PATH}"
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def preprocess_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Run the complete preprocessing pipeline."""

    df = clean_column_names(df)

    validate_columns(df)

    df = remove_duplicates(df)

    df = clean_datetime(df)

    df = clean_numeric_columns(df)

    df = handle_missing_values(df)

    df = clean_categorical_columns(df)

    df = handle_invalid_values(df)

    df = sort_by_datetime(df)

    return df


def main() -> None:
    """Run preprocessing pipeline."""

    print("=" * 60)
    print("TRAFFIC DATA PREPROCESSING")
    print("=" * 60)

    df = load_data()

    print("\nStarting preprocessing...")

    cleaned_df = preprocess_data(df)

    print("\nPreprocessing completed.")

    print(
        f"Original rows: {len(df):,}"
    )

    print(
        f"Cleaned rows: {len(cleaned_df):,}"
    )

    print(
        f"Rows removed: "
        f"{len(df) - len(cleaned_df):,}"
    )

    print(
        f"Remaining missing values: "
        f"{cleaned_df.isna().sum().sum()}"
    )

    save_data(cleaned_df)


if __name__ == "__main__":
    main()