"""Serving-side tests for the second-stage LightGBM ranker."""

import numpy as np

from app.config import get_settings
from app.db import SessionLocal
from app.models import ModelBlob
from app.services.recommender import get_service, load_reranker, reset_service


def test_ranker_is_loaded_and_used(client, auth):
    with SessionLocal() as db:
        service = get_service(db)
    assert service.reranker is not None, "the seeded ranker should be loaded into the recommender"

    fantasy = client.get("/books/search", params={"q": "fantasy book", "limit": 3}).json()
    client.post("/me/onboarding", json={"loved_book_ids": [b["id"] for b in fantasy]}, headers=auth)
    recs = client.get("/me/recommendations", params={"limit": 10, "explore": False}, headers=auth).json()
    assert len(recs["items"]) == 10
    assert not {r["book"]["id"] for r in recs["items"]} & {b["id"] for b in fantasy}


def test_ranker_can_be_disabled(client, auth, monkeypatch):
    fantasy = client.get("/books/search", params={"q": "fantasy book", "limit": 3}).json()
    client.post("/me/onboarding", json={"loved_book_ids": [b["id"] for b in fantasy]}, headers=auth)
    with_ranker = client.get("/me/recommendations", params={"limit": 10, "explore": False}, headers=auth).json()

    monkeypatch.setattr(get_settings(), "recommendation_use_ranker", False)
    stage1_only = client.get("/me/recommendations", params={"limit": 10, "explore": False}, headers=auth).json()
    assert len(stage1_only["items"]) == len(with_ranker["items"]) == 10


def test_unloadable_ranker_falls_back_to_stage_one(client, auth, caplog, monkeypatch):
    """A model that fails to load must never take the API down - it serves stage-1 ranking."""
    import lightgbm as lgb

    def boom(*args, **kwargs):
        raise lgb.basic.LightGBMError("Model format error")

    monkeypatch.setattr(lgb, "Booster", boom)
    with SessionLocal() as db:
        assert load_reranker(db) is None
    assert "could not load" in caplog.text.lower()

    reset_service()
    try:
        res = client.get("/me/recommendations", params={"limit": 5}, headers=auth)
        assert res.status_code == 200 and len(res.json()["items"]) == 5
    finally:
        monkeypatch.undo()
        reset_service()


def test_ranker_blob_is_stored_with_unix_newlines(client):
    """Regression: a CRLF model file made LightGBM abort the process on load."""
    with SessionLocal() as db:
        data = db.get(ModelBlob, "ranker").data
    assert b"\r\n" not in data and data.startswith(b"tree")


def test_reranker_scores_have_no_nans(client):
    """Guard against features (e.g. series_number) producing NaN scores for real catalog rows."""
    from alexandria_core import FEATURE_NAMES, rerank_features

    with SessionLocal() as db:
        service = get_service(db)
    rec = service.recommender
    feedback = {0: 2.0, 5: 1.0, 7: -1.0}
    blend = rec.blend(feedback, [])
    pool = rec.candidates(blend, feedback, exclude=[], size=50)
    scores = service.reranker.score(rec, blend, feedback, [], pool)
    assert scores.shape == (len(pool),) and np.isfinite(scores).all()
    # series_number is deliberately NaN for standalone books (LightGBM handles missing values).
    features = rerank_features(rec, blend, feedback, [], pool)
    finite_columns = [i for i, name in enumerate(FEATURE_NAMES) if name != "series_number"]
    assert np.isfinite(features[:, finite_columns]).all()
