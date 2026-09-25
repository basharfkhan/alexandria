"""Offline evaluation: compare baselines against the production hybrid recommender.

Models compared (all on the same held-out positives, excluding each user's train items):

* popularity         - most-liked books overall (the baseline any recommender must beat)
* content            - mean embedding of the user's liked books
* bpr                - BPR-MF with the user embedding learned during training
* hybrid_foldin      - production path: HybridRecommender with fold-in (no learned user vector)
* hybrid_cold5       - production path given only 5 of the user's ratings (new-user scenario)
* hybrid_served      - stage 1 as the API returns it: + MMR diversity and a 3-books-per-author cap
* rerank_*           - the same paths with the learned second-stage ranker (when one is trained)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from alexandria_core import CatalogArrays, HybridRecommender
from alexandria_core.metrics import evaluate_rankings
from alexandria_ml.config import POSITIVE_RATING
from alexandria_ml.models.bpr import BPRMF

log = logging.getLogger(__name__)

# Must match the API's settings (recommendation_diversity / recommendation_max_per_author).
SERVED_DIVERSITY = 0.25
SERVED_AUTHOR_CAP = 3

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
        authors=books.authors.tolist(),
        titles=books.title.tolist(),
    )


@dataclass
class EvalContext:
    """The sampled users plus their train/test items, shared by every model under comparison."""

    users: np.ndarray
    test_pos: dict[int, set[int]]
    train_by_user: dict[int, pd.DataFrame]
    seen: dict[int, set[int]]
    n_items: int
    k: int

    def run(self, name: str, rank_user: Callable[[int, int], list[int]]) -> dict[str, float]:
        rankings = {int(u): rank_user(int(u), row) for row, u in enumerate(self.users)}
        metrics = evaluate_rankings(rankings, self.test_pos, self.n_items, ks=(10, self.k))
        log.info("%-15s %s", name, "  ".join(f"{m}={v:.4f}" for m, v in metrics.items() if m != "n_users"))
        return metrics


def build_context(
    train: pd.DataFrame, test: pd.DataFrame, n_items: int, k: int = 20, max_users: int = 2000, seed: int = 42
) -> EvalContext:
    rng = np.random.default_rng(seed)
    test_pos = test[test.rating >= POSITIVE_RATING].groupby("user_idx").item_idx.apply(set).to_dict()
    users = np.array(sorted(test_pos))
    if len(users) > max_users:
        users = np.sort(rng.choice(users, size=max_users, replace=False))
    train_by_user = {int(u): g for u, g in train[train.user_idx.isin(users)].groupby("user_idx")}
    return EvalContext(
        users=users,
        test_pos={int(u): test_pos[u] for u in users},
        train_by_user=train_by_user,
        seen={u: set(g.item_idx.tolist()) for u, g in train_by_user.items()},
        n_items=n_items,
        k=k,
    )


def hybrid_metrics(
    hybrid: HybridRecommender, ctx: EvalContext, name: str, n_ratings: int | None = None, **recommend_kwargs
) -> dict[str, float]:
    """Evaluate the serving path; ``n_ratings`` keeps only that many random ratings per user."""

    def rank(u: int, _row: int) -> list[int]:
        ratings = ctx.train_by_user[u]
        if n_ratings is not None and len(ratings) > n_ratings:
            ratings = ratings.sample(n=n_ratings, random_state=u)
        kwargs = {"diversity": 0.0, **recommend_kwargs}
        recs = hybrid.recommend(feedback_from_ratings(ratings), k=ctx.k, exclude=ctx.seen[u], **kwargs)
        return [r.index for r in recs]

    return ctx.run(name, rank)


def evaluate_models(
    books: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    content: np.ndarray,
    model: BPRMF,
    k: int = 20,
    max_users: int = 2000,
    seed: int = 42,
    hybrid_kwargs: dict | None = None,
    reranker=None,
) -> dict[str, dict[str, float]]:
    ctx = build_context(train, test, len(books), k=k, max_users=max_users, seed=seed)

    factors, bias = model.item_factors()
    hybrid = HybridRecommender(catalog_arrays(books, content, factors, bias), **(hybrid_kwargs or {}))
    content_norm = hybrid.content

    pos_train = train[train.rating >= POSITIVE_RATING]
    popularity = np.bincount(pos_train.item_idx, minlength=len(books)).astype(np.float64)

    with torch.no_grad():
        bpr_scores = model.all_scores(torch.as_tensor(ctx.users, dtype=torch.long)).numpy()

    def content_rank(u: int, _row: int) -> list[int]:
        liked = ctx.train_by_user[u]
        liked = liked[liked.rating >= POSITIVE_RATING].item_idx.to_numpy()
        if len(liked) == 0:
            return _top_k(popularity, ctx.seen[u], k)
        return _top_k(content_norm @ content_norm[liked].mean(axis=0), ctx.seen[u], k)

    served = {"diversity": SERVED_DIVERSITY, "max_per_author": SERVED_AUTHOR_CAP}
    results = {
        "popularity": ctx.run("popularity", lambda u, _: _top_k(popularity, ctx.seen[u], k)),
        "content": ctx.run("content", content_rank),
        "bpr": ctx.run("bpr", lambda u, row: _top_k(bpr_scores[row], ctx.seen[u], k)),
        "hybrid_foldin": hybrid_metrics(hybrid, ctx, "hybrid_foldin"),
        "hybrid_cold5": hybrid_metrics(hybrid, ctx, "hybrid_cold5", n_ratings=5),
        "hybrid_served": hybrid_metrics(hybrid, ctx, "hybrid_served", **served),
    }
    if reranker is not None:
        results["rerank_foldin"] = hybrid_metrics(hybrid, ctx, "rerank_foldin", reranker=reranker)
        results["rerank_cold5"] = hybrid_metrics(hybrid, ctx, "rerank_cold5", n_ratings=5, reranker=reranker)
        results["rerank_served"] = hybrid_metrics(hybrid, ctx, "rerank_served", reranker=reranker, **served)
    return results
