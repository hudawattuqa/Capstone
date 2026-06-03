from typing import Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    MODEL_PATH    : str = "saved_model/placement_model.keras"
    SCALER_PATH   : str = "saved_model/placement_scaler.pkl"
    FEATURES_PATH : str = "saved_model/feature_names.txt"
    MODEL_META_PATH: str = "saved_model/placement_model_metadata.json"

    HOST : str = "0.0.0.0"
    PORT : int = 8000

    ALLOWED_ORIGINS: list[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

settings = Settings()
