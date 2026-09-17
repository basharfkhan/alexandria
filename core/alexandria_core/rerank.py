"""Second-stage learning-to-rank features.

The hybrid recommender (stage 1) is a hand-tuned linear blend. Stage 2 lets a learned model
(LightGBM LambdaMART, trained in ml/alexandria_ml/ranker.py) reorder the top candidates using
richer, non-linear evidence: agreement between signals, similarity to *individual* liked books,
author and series continuity, and how much we know about the reader.

Features are computed here, in the numpy-only core package, so training and serving build the
exact same matrix. The model itself is injected: anything with ``predict(X) -> scores``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from alexandria_core.recommender import Blend, HybridRecommender

FEATURE_NAMES = [
    # first-stage scores
    "stage1_score", "stage1_rank_pct",
    "part_content", "part_cf", "part_genre", "part_popularity", "part_quality",
    "content_cosine", "cf_score", "cf_score_pool_z", "content_cosine_pool_z",
    # relationship to the reader's liked / disliked books
    "max_content_sim_liked", "mean_content_sim_liked", "max_cf_sim_liked", "max_content_sim_disliked",
    "same_author_liked", "same_author_disliked", "same_series_liked", "next_in_series",
    "series_number", "genre_overlap_liked",
    # item
    "popularity_z", "quality_z", "cf_item_bias", "n_genres",
    # reader
    "n_liked", "n_disliked", "cf_weight", "n_selected_genres",
]


class _Model(Protocol):
    def predict(self, x: np.ndarray) -> np.ndarray: ...


def _zscore(x: np.ndarray) -> np.ndarray:
    std = x.std()
    return (x - x.mean()) / std if std > 1e-9 else np.zeros_like(x)


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def rerank_features(
    rec: HybridRecommender,
    blend: Blend,
    feedback: Mapping[int, float],
    genres: Sequence[str],
    pool: np.ndarray,
) -> np.ndarray:
    """Feature matrix of shape (len(pool), len(FEATURE_NAMES)); ``pool`` is ordered best-first."""
    n = len(pool)
    liked = np.array([i for i, w in feedback.items() if w > 0], dtype=np.int64)
    disliked = np.array([i for i, w in feedback.items() if w < 0], dtype=np.int64)
    parts, raw = blend.parts, blend.raw

    content_cos = np.asarray(raw["content"], dtype=np.float64)[pool]
    cf_score = np.asarray(raw["cf"], dtype=np.float64)[pool]

    cand_content = rec.content[pool]
    if len(liked):
        sims = cand_content @ rec.content[liked].T
        max_sim_liked, mean_sim_liked = sims.max(axis=1), sims.mean(axis=1)
    else:
        max_sim_liked = mean_sim_liked = np.zeros(n)
    max_sim_disliked = (cand_content @ rec.content[disliked].T).max(axis=1) if len(disliked) else np.zeros(n)

    if rec.has_cf and len(liked):
        max_cf_sim = (_normalize(rec.cf_factors[pool]) @ _normalize(rec.cf_factors[liked]).T).max(axis=1)
    else:
        max_cf_sim = np.zeros(n)

    # Author and series continuity.
    authors = rec.item_author or [None] * rec.n_items
    liked_authors: dict[str, int] = {}
    disliked_authors: dict[str, int] = {}
    for i in liked:
        liked_authors[authors[i]] = liked_authors.get(authors[i], 0) + 1
    for i in disliked:
        disliked_authors[authors[i]] = disliked_authors.get(authors[i], 0) + 1
    liked_series: dict[str, list[float]] = {}
    for i in liked:
        if rec.item_series[i]:
            liked_series.setdefault(rec.item_series[i], []).append(rec.item_series_no[i])

    same_author_liked = np.array([liked_authors.get(authors[i], 0) for i in pool], dtype=np.float64)
    same_author_disliked = np.array([disliked_authors.get(authors[i], 0) for i in pool], dtype=np.float64)
    same_series_liked = np.zeros(n)
    next_in_series = np.zeros(n)
    for row, i in enumerate(pool):
        series = rec.item_series[i]
        if series and series in liked_series:
            numbers = liked_series[series]
            same_series_liked[row] = len(numbers)
            next_in_series[row] = float(rec.item_series_no[i] == max(numbers) + 1)

    # Genre overlap with the genres of liked books.
    liked_genres: dict[str, int] = {}
    for i in liked:
        for g in rec.item_genres[i]:
            liked_genres[g] = liked_genres.get(g, 0) + 1
    genre_overlap = np.array(
        [sum(liked_genres.get(g, 0) for g in rec.item_genres[i]) / max(len(liked), 1) for i in pool]
    )

    total = blend.total[pool]
    x = np.column_stack([
        total, np.arange(n) / max(n - 1, 1),
        parts["content"][pool], parts["cf"][pool], parts["genre"][pool],
        parts["popularity"][pool], parts["quality"][pool],
        content_cos, cf_score, _zscore(cf_score), _zscore(content_cos),
        max_sim_liked, mean_sim_liked, max_cf_sim, max_sim_disliked,
        same_author_liked, same_author_disliked, same_series_liked, next_in_series,
        rec.item_series_no[pool], genre_overlap,
        rec.pop_z[pool], rec.quality_z[pool], rec.cf_bias[pool],
        np.array([len(rec.item_genres[i]) for i in pool], dtype=np.float64),
        np.full(n, len(liked)), np.full(n, len(disliked)), np.full(n, float(raw["w_cf"])),
        np.full(n, len(genres)),
    ])
    assert x.shape[1] == len(FEATURE_NAMES)
    return x


class Reranker:
    """Wraps a trained ranking model; checks it was trained on this exact feature set."""

    def __init__(self, model: _Model, feature_names: Sequence[str] | None = None):
        if feature_names is not None and list(feature_names) != FEATURE_NAMES:
            raise ValueError("ranker was trained on a different feature set - retrain it")
        self.model = model

    def score(
        self,
        rec: HybridRecommender,
        blend: Blend,
        feedback: Mapping[int, float],
        genres: Sequence[str],
        pool: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(self.model.predict(rerank_features(rec, blend, feedback, genres, pool)), dtype=np.float64)
