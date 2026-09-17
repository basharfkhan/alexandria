"""Offline evaluation: compare baselines against the production hybrid recommender.

Models compared (all on the same held-out positives, excluding each user's train items):

* popularity         - most-liked books overall (the baseline any recommender must beat)
* content            - mean embedding of the user's liked books
* bpr                - BPR-MF with the user embedding learned during training
* hybrid_foldin      - production path: HybridRecommender with fold-in (no learned user vector)
* hybrid_cold5       - production path given only 5 of the user's ratings (new-user scenario)
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch

from alexandria_core import CatalogArrays, HybridRecommender
from alexandria_core.metrics import evaluate_rankings
from alexandria_ml.config import POSITIVE_RATING
from alexandria_ml.models.bpr import BPRMF

log = logging.getLogger(__name__)

RATING_TO_WEIGHT = {5: 2.0, 4: 1.0, 3: 0.0, 2: -1.0, 1: -1.0}


def feedback_from_ratings(user_ratings: pd.DataFrame) -> dict[int, float]:
    weights = user_ratings.rating.map(RATING_TO_WEIGHT)
    return {int(i): float(w) for i, w in zip(user_ratings.item_idx, weights, strict=True) if w != 0}


def _top_k(scores: np.ndarray, blocked: set[int], k: int) -> list[int]:
    scores = scores.astype(np.float64, copy=True)
    if blocked:
        scores[list(blocked)] = -np.inf
    top = np.argpartition(-scores, k)[:k]
    return top[np.argsort(-scores[top])].tolist()


def catalog_arrays(books: pd.DataFrame, content: np.ndarray, factors=None, bias=None) -> CatalogArrays:
    return CatalogArrays(
        content=content,
        popularity=books.ratings_count.to_numpy(),
        avg_rating=books.avg_rating.to_numpy(),
        genres=books.genres.tolist(),
        cf_factors=factors,
        cf_bias=bias,
    )


def evaluate_models(
    books: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    content: np.ndarray,
    model: BPRMF,
    k: int = 20,
    max_users: int = 2000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    n_items = len(books)

    test_pos = test[test.rating >= POSITIVE_RATING].groupby("user_idx").item_idx.apply(set).to_dict()
    users = np.array(sorted(test_pos))
    if len(users) > max_users:
        users = np.sort(rng.choice(users, size=max_users, replace=False))
    test_pos = {int(u): test_pos[u] for u in users}

    train_by_user = {int(u): g for u, g in train[train.user_idx.isin(users)].groupby("user_idx")}
    seen = {u: set(g.item_idx.tolist()) for u, g in train_by_user.items()}

    factors, bias = model.item_factors()
    hybrid = HybridRecommender(catalog_arrays(books, content, factors, bias))
    content_norm = hybrid.content

    pos_train = train[train.rating >= POSITIVE_RATING]
    popularity = np.bincount(pos_train.item_idx, minlength=n_items).astype(np.float64)

    with torch.no_grad():
        bpr_scores = model.all_scores(torch.as_tensor(users, dtype=torch.long)).numpy()

    def run(name: str, rank_user: Callable[[int, int], list[int]]) -> dict[str, float]:
        rankings = {int(u): rank_user(int(u), row) for row, u in enumerate(users)}
        metrics = evaluate_rankings(rankings, test_pos, n_items, ks=(10, k))
        log.info("%-15s %s", name, "  ".join(f"{m}={v:.4f}" for m, v in metrics.items() if m != "n_users"))
        return metrics

    def content_rank(u: int, _row: int) -> list[int]:
        liked = train_by_user[u]
        liked = liked[liked.rating >= POSITIVE_RATING].item_idx.to_numpy()
        if len(liked) == 0:
            return _top_k(popularity, seen[u], k)
        profile = content_norm[liked].mean(axis=0)
        return _top_k(content_norm @ profile, seen[u], k)

    def hybrid_rank(u: int, _row: int, n_ratings: int | None = None) -> list[int]:
        ratings = train_by_user[u]
        if n_ratings is not None and len(ratings) > n_ratings:
            ratings = ratings.sample(n=n_ratings, random_state=u)
        recs = hybrid.recommend(feedback_from_ratings(ratings), k=k, exclude=seen[u], diversity=0.0)
        return [r.index for r in recs]

    return {
        "popularity": run("popularity", lambda u, _: _top_k(popularity, seen[u], k)),
        "content": run("content", content_rank),
        "bpr": run("bpr", lambda u, row: _top_k(bpr_scores[row], seen[u], k)),
        "hybrid_foldin": run("hybrid_foldin", hybrid_rank),
        "hybrid_cold5": run("hybrid_cold5", lambda u, row: hybrid_rank(u, row, n_ratings=5)),
    }
