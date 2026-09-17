import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=Path(__file__).resolve().parents[1] / ".env", extra="ignore")

    database_url: str = "postgresql+psycopg://alexandria:alexandria@localhost:5432/alexandria"

    jwt_secret: str = "dev-secret-change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24 * 7

    # Accepts a JSON list or a comma-separated string: "https://a.vercel.app,http://localhost:3000"
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:3000"]

    # Vector widths - must match the artifacts produced by the ML pipeline.
    content_dim: int = 384
    cf_dim: int = 64

    anthropic_api_key: str | None = None
    llm_model: str = "claude-opus-5"
    llm_effort: str = "low"

    recommendation_explore_slots: int = 2
    # Keep in sync with SERVED_* in ml/alexandria_ml/evaluate.py so offline metrics describe what users see.
    recommendation_diversity: float = 0.25
    recommendation_max_per_author: int = 3

    @field_validator("database_url")
    @classmethod
    def use_psycopg_driver(cls, url: str) -> str:
        """Hosts (Neon, Render, Heroku) hand out postgres:// or postgresql:// URLs; SQLAlchemy needs the driver named."""
        return re.sub(r"^postgres(ql)?://", "postgresql+psycopg://", url.strip())

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            value = value.strip()
            value = json.loads(value) if value.startswith("[") else value.split(",")
        return [origin.strip().rstrip("/") for origin in value if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
