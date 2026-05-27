"""
Konfigurasi aplikasi — semua nilai bisa di-override lewat environment variable
atau file .env.

Contoh .env:
    MODEL_PATH=saved_model/placement_model.keras
    HOST=0.0.0.0
    PORT=8000
    ALLOWED_ORIGINS=["http://localhost:3000","https://api.mathquest.id"]
"""

from typing import Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Path model ────────────────────────────────────────────────────────────
    MODEL_PATH    : str = "saved_model/placement_model.keras"
    SCALER_PATH   : str = "saved_model/placement_scaler.pkl"
    FEATURES_PATH : str = "saved_model/feature_names.txt"
    MODEL_META_PATH: str = "saved_model/placement_model_metadata.json"

    # ── Server ────────────────────────────────────────────────────────────────
    HOST : str = "0.0.0.0"
    PORT : int = 8000

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Bisa diisi dengan list domain frontend/backend yang boleh akses
    ALLOWED_ORIGINS: list[str] = ["*"]

    # ── Pydantic Settings ─────────────────────────────────────────────────────
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
