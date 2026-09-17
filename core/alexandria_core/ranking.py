"""Score normalisation and re-ranking utilities."""

from __future__ import annotations

import numpy as np


def zscore(x: np.ndarray) -> np.ndarray:
    """Standardise scores so signals on different scales can be blended linearly."""
    x = np.asarray(x, dtype=np.float64)
    std = x.std()
    if std < 1e-9:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def cf_weight(n_explicit: int, max_weight: float = 0.6, half_life: float = 5.0) -> float:
    """How much to trust collaborative filtering given how much a user has told us.

    A new user with one rating gets mostly content-based recommendations; as feedback
    accumulates the latent-factor signal takes over (saturating at ``max_weight``).
    """
    if n_explicit <= 0:
        return 0.0
    return max_weight * n_explicit / (n_explicit + half_life)


def mmr_rerank(
    candidates: np.ndarray,
    relevance: np.ndarray,
    embeddings: np.ndarray,
    k: int,
    lambda_: float = 0.75,
) -> np.ndarray:
    """Maximal Marginal Relevance: trade relevance against redundancy.

    Args:
        candidates: item indices, pre-sorted or not.
        relevance: relevance score per candidate (same order as ``candidates``).
        embeddings: L2-normalised item embeddings for the full catalog.
        k: number of items to select.
        lambda_: 1.0 = pure relevance, 0.0 = pure diversity.

    Returns:
        Selected item indices in rank order.
    """
    candidates = np.asarray(candidates)
    k = min(k, len(candidates))
    if k == 0:
        return candidates[:0]

    rel = np.asarray(relevance, dtype=np.float64)
    span = rel.max() - rel.min()
    rel = (rel - rel.min()) / span if span > 1e-9 else np.ones_like(rel)

    cand_emb = embeddings[candidates]
    max_sim = np.zeros(len(candidates))
    available = np.ones(len(candidates), dtype=bool)
    chosen: list[int] = []

    for step in range(k):
        mmr = lambda_ * rel - (1.0 - lambda_) * (max_sim if step else 0.0)
        mmr = np.where(available, mmr, -np.inf)
        pick = int(np.argmax(mmr))
        chosen.append(pick)
        available[pick] = False
        max_sim = np.maximum(max_sim, cand_emb @ cand_emb[pick])

    return candidates[chosen]
