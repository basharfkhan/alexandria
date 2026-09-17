from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=Path(__file__).resolve().parents[1] / ".env", extra="ignore")

    database_url: str = "postgresql+psycopg://alexandria:alexandria@localhost:5432/alexandria"

    jwt_secret: str = "dev-secret-change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24 * 7

    cors_origins: list[str] = ["http://localhost:3000"]

    # Vector widths - must match the artifacts produced by the ML pipeline.
    content_dim: int = 384
    cf_dim: int = 64

    anthropic_api_key: str | None = None
    llm_model: str = "claude-opus-5"
    llm_effort: str = "low"

    recommendation_explore_slots: int = 2


@lru_cache
def get_settings() -> Settings:
    return Settings()
