import pytest

from app.config import Settings


@pytest.mark.parametrize(
    "given",
    [
        "postgresql://u:p@ep-x.neon.tech/db?sslmode=require",
        "postgres://u:p@ep-x.neon.tech/db?sslmode=require",
        "postgresql+psycopg://u:p@ep-x.neon.tech/db?sslmode=require",
    ],
)
def test_database_url_gets_psycopg_driver(given):
    assert Settings(database_url=given).database_url == "postgresql+psycopg://u:p@ep-x.neon.tech/db?sslmode=require"


def test_sqlite_url_untouched():
    assert Settings(database_url="sqlite:///./x.db").database_url == "sqlite:///./x.db"


@pytest.mark.parametrize(
    "given",
    ['["https://a.vercel.app/", "http://localhost:3000"]', "https://a.vercel.app/, http://localhost:3000"],
)
def test_cors_origins_json_or_comma_separated(given, monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", given)
    assert Settings().cors_origins == ["https://a.vercel.app", "http://localhost:3000"]
