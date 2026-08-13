"""
Application configuration.
"""

from pathlib import Path

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)


class Settings(BaseSettings):
    """Application settings."""

    app_name: str = "Traffic Prediction API"

    app_version: str = "1.0.0"

    debug: bool = False

    model_path: str = (
        "models/traffic_volume_timeseries.joblib"
    )

    database_url: str

    jwt_secret_key: str

    jwt_algorithm: str = "HS256"

    access_token_expire_minutes: int = 120

    # Modern Pydantic V2 Configuration Style
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


settings = Settings()
