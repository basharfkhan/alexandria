"""Loads the catalog's vectors from the database into an in-memory HybridRecommender.

With a 10k-book catalog the full matrices are ~20MB, so brute-force numpy scoring
over every book takes a few milliseconds. Postgres/pgvector stays the source of truth
(and serves nearest-neighbour "similar books" queries); at millions of items the
candidate-generation step would move into the vector index instead.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from alexandria_core import FEEDBACK_WEIGHTS, CatalogArrays, HybridRecommender, Recommendation, Reranker
from app.config import get_settings
from app.models import Book, Interaction, ModelBlob, ModelMeta, User


@dataclass
class RecommenderService:
    recommender: HybridRecommender
    book_ids: np.ndarray  # catalog index -> book id
    index_of: dict[int, int]  # book id -> catalog index
    model_version: str | None
    reranker: Reranker | None = None

    def user_feedback(self, interactions: list[Interaction]) -> dict[int, float]:
        return {
            self.index_of[it.book_id]: FEEDBACK_WEIGHTS[it.signal]
            for it in interactions
            if it.book_id in self.index_of
        }

    def recommend_for(self, user: User, k: int, explore_slots: int) -> tuple[list[Recommendation], float]:
        feedback = self.user_feedback(user.interactions)
        # Roughly one exploration pick per eight recommendations, so short lists stay focused.
        settings = get_settings()
        reranker = self.reranker if settings.recommendation_use_ranker else None
        recs = self.recommender.recommend(
            feedback,
            genres=user.favorite_genres or [],
            k=k,
            explore_slots=min(explore_slots, k // 8),
            diversity=settings.recommendation_diversity,
            max_per_author=settings.recommendation_max_per_author,
            reranker=reranker,
        )
        n_explicit = sum(abs(w) >= 1 for w in feedback.values())
        return recs, min(1.0, n_explicit / 10)


log = logging.getLogger(__name__)

_lock = threading.Lock()
_service: RecommenderService | None = None
_last_version_check = 0.0


def load_service(db: Session) -> RecommenderService:
    rows = db.execute(
        select(
            Book.id, Book.content_embedding, Book.cf_factors, Book.cf_bias,
            Book.ratings_count, Book.avg_rating, Book.genres, Book.authors, Book.title,
        ).order_by(Book.id)
    ).all()
    if not rows:
        raise LookupError("The book catalog is empty - run `python -m app.seed` first.")

    has_cf = all(r.cf_factors is not None for r in rows)
    catalog = CatalogArrays(
        content=np.vstack([np.asarray(r.content_embedding, dtype=np.float32) for r in rows]),
        popularity=np.array([r.ratings_count for r in rows]),
        avg_rating=np.array([r.avg_rating for r in rows]),
        genres=[r.genres or [] for r in rows],
        cf_factors=np.vstack([np.asarray(r.cf_factors, dtype=np.float32) for r in rows]) if has_cf else None,
        cf_bias=np.array([r.cf_bias or 0.0 for r in rows]) if has_cf else None,
        authors=[r.authors for r in rows],
        titles=[r.title for r in rows],
    )
    ids = np.array([r.id for r in rows])
    meta = db.get(ModelMeta, "manifest")
    return RecommenderService(
        recommender=HybridRecommender(catalog),
        book_ids=ids,
        index_of={int(b): i for i, b in enumerate(ids)},
        model_version=(meta.value or {}).get("model_version") if meta else None,
        reranker=load_reranker(db),
    )


def load_reranker(db: Session) -> Reranker | None:
    """Load the seeded LightGBM ranker, if any. A stale or unreadable model is skipped, not fatal."""
    blob = db.get(ModelBlob, "ranker")
    if blob is None:
        return None
    try:
        import lightgbm as lgb

        booster = lgb.Booster(model_str=blob.data.decode("utf-8").replace("\r\n", "\n"))
        reranker = Reranker(booster, booster.feature_name())
    except Exception:  # noqa: BLE001 - never let a bad ranker take the API down
        log.exception("could not load the second-stage ranker; serving stage-1 ranking only")
        return None
    log.info("loaded second-stage ranker (%d trees)", booster.num_trees())
    return reranker


def _seeded_version(db: Session) -> str | None:
    meta = db.get(ModelMeta, "manifest")
    return (meta.value or {}).get("model_version") if meta else None


def get_service(db: Session) -> RecommenderService:
    """Return the in-memory recommender, hot-reloading it when a new model version is seeded.

    Seeding writes the manifest last, so a version change means the catalog is fully updated.
    The check is one primary-key lookup, rate limited by ``model_reload_interval_s``.
    """
    global _service, _last_version_check
    if _service is None:
        with _lock:
            if _service is None:
                _service = load_service(db)
                _last_version_check = time.monotonic()
        return _service

    if time.monotonic() - _last_version_check >= get_settings().model_reload_interval_s:
        with _lock:
            _last_version_check = time.monotonic()
            version = _seeded_version(db)
            if version != _service.model_version:
                log.info("model version changed %s -> %s; reloading recommender", _service.model_version, version)
                _service = load_service(db)
    return _service


def reset_service() -> None:
    global _service
    with _lock:
        _service = None
