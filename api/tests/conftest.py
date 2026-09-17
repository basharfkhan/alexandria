"""Test fixtures: SQLite database seeded with a small synthetic artifact set.

Settings are read at import time, so the environment is configured before `app` is imported.
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

_tmp = Path(tempfile.mkdtemp(prefix="alexandria-test-"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_tmp / 'test.db').as_posix()}"
os.environ["JWT_SECRET"] = "test-secret-that-is-at-least-32-bytes-long"
os.environ.pop("ANTHROPIC_API_KEY", None)

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.seed import seed  # noqa: E402

GENRES = ["fantasy", "romance", "mystery"]


def make_artifacts(out: Path, n: int = 90) -> Path:
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(3, 384))
    cf_centers = rng.normal(size=(3, 64))
    labels = np.arange(n) % 3
    books = [
        {
            "book_id": i + 1,
            "title": f"{GENRES[labels[i]].title()} Book {i + 1}",
            "authors": f"Writer {labels[i]}",
            "year": 2000 + i % 20,
            "avg_rating": 3.5 + (i % 10) / 10,
            "ratings_count": 1000 + 37 * i,
            "image_url": None,
            "genres": [GENRES[labels[i]]],
            "tags": [GENRES[labels[i]]],
        }
        for i in range(n)
    ]
    out.mkdir(parents=True, exist_ok=True)
    (out / "books.json").write_text(json.dumps(books))
    np.save(out / "content_embeddings.npy", centers[labels] + 0.2 * rng.normal(size=(n, 384)))
    np.save(out / "cf_factors.npy", cf_centers[labels] + 0.2 * rng.normal(size=(n, 64)))
    np.save(out / "cf_bias.npy", np.zeros(n))
    (out / "manifest.json").write_text(json.dumps({"model_version": "test-1"}))
    return out


@pytest.fixture(scope="session")
def client():
    seed(make_artifacts(_tmp / "artifacts"))
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth(client):
    """Register a fresh user and return auth headers."""
    import uuid

    res = client.post("/auth/register", json={"username": f"u{uuid.uuid4().hex[:10]}", "password": "password123"})
    assert res.status_code == 201, res.text
    return {"Authorization": f"Bearer {res.json()['access_token']}"}
