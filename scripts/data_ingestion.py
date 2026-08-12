"""
Traffic Prediction System
Data Ingestion Module
"""

from pathlib import Path

import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data paths
RAW_DATA_DIR = PROJECT_ROOT / "ml" / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "ml" / "data" / "processed"

RAW_FILE = RAW_DATA_DIR / "Metro_Interstate_Traffic_Volume.csv"
PROCESSED_FILE = PROCESSED_DATA_DIR / "traffic_data.csv"


def load_raw_data() -> pd.DataFrame:
    """Load the raw traffic dataset."""

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_FILE}"
        )

    df = pd.read_csv(RAW_FILE)

    return df


def validate_data(df: pd.DataFrame) -> None:
    """Perform basic dataset validation."""

    if df.empty:
        raise ValueError("Dataset is empty.")

    required_columns = {
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

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )


def save_processed_data(df: pd.DataFrame) -> None:
    """Save the dataset to the processed directory."""

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        PROCESSED_FILE,
        index=False
    )


def main() -> None:
    """Run the complete ingestion process."""

    print("=" * 60)
    print("TRAFFIC DATA INGESTION")
    print("=" * 60)

    print("\nLoading dataset...")

    df = load_raw_data()

    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    print("\nValidating dataset...")

    validate_data(df)

    print("Validation successful.")

    print("\nSaving processed dataset...")

    save_processed_data(df)

    print(f"Saved to: {PROCESSED_FILE}")

    print("\nData ingestion completed successfully.")


if __name__ == "__main__":
    main()